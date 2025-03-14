#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

'''


import open3d as o3d
import numpy as np
import cv2 as cv
import json
import random
import math
import os
import re
import socket
import json
import time
import pyrealsense2 as rs
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as SR
from matplotlib import pyplot as plt
from threading import Thread



np.set_printoptions(threshold=np.inf)

'''全局参数设置'''
'''设置输入参数: 模板名称 与 序号'''
model_name='ambrosial'
model_select=1

"FLANN算法的筛选比例"
match_ratio=0.7

"实际相机参数"
width_resolution=1280
height_resolution=720
fx_camera=921.305
fy_camera=921.583
cx_camera=648.475
cy_camera=359.169

'''由于模板构建时没有确定好中心位置高度，所以在此需要手动调节'''
model_bias_T=np.array([[1.0,0.0,0.0,0.0],
                       [0.0,1.0,0.0,0.0],
                       [0.0,0.0,1.0,-0.02],
                       [0.0,0.0,0.0,1.0]])


'''模板选择'''
current_path=os.path.abspath(__file__)
current_path=os.path.dirname(current_path)

model2camera_parameters_path=os.path.join(current_path,'models',model_name,'test.txt')
model2camera_parameter_name=os.path.join('select{}'.format(str(model_select)))
model_color_path=os.path.join(current_path,'models',model_name,'select{}_color_save.png'.format(str(model_select)))
model_depth_path=os.path.join(current_path,'models',model_name,'select{}_depth_save.png'.format(str(model_select)))




#设置最低匹配数量为10
MIN_MATCH_COUNT = 6
#设置ICP算法所用到的求解点数量，注意不要太少，否则容易退化，导致求解得到的变换矩阵不正交
model_solve_points=6



'''ICP求解'''
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform between corresponding 3D points A->B
    Input:
      A: Nx3 numpy array of corresponding 3D points
      B: Nx3 numpy array of corresponding 3D points
    Returns:
      T: 4x4 homogeneous transformation matrix
      R: 3x3 rotation matrix
      t: 3x1 column vector
    '''

    assert len(A) == len(B)

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B


    # rotation matrix
    W = np.dot(BB.T, AA)
    U, s, VT = np.linalg.svd(W)
    R = np.dot(U, VT)

    # special reflection case
    if np.linalg.det(R) < 0:
        VT[2, :] *= -1
        R = np.dot(U, VT)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T, R, t




'''模板图像 深度 转 三维坐标'''
def Depth2xyz(u,v,depthValue):
  #模板的采集相机内参数
  fx=1949.0157470703125
  fy=1948.34521484375
  cx=2053.27197265625
  cy=1556.284912109375
  
  depthValue=np.array(depthValue)
  u=np.array(u)
  v=np.array(v)

  '''转换单位 m'''
  z=depthValue*0.001
  x=((u-cx)*z)/fx
  y=((v-cy)*z)/fy


  result=[x,y,z]
  result=np.array(result)
  result=np.transpose(result)

  # print("result")
  # print(result.shape)
  
  return result




'''实际图像 深度 转 三维坐标'''
def Depth2xyz2(u,v,depthValue):
  #实际图像的相机内参数
  fx=921.305
  fy=921.583
  cx=648.475
  cy=359.169
      
  depthValue=np.array(depthValue)
  u=np.array(u)
  v=np.array(v)

  '''转换单位 m'''
  z=depthValue*0.001
  x=((u-cx)*z)/fx
  y=((v-cy)*z)/fy


  result=[x,y,z]
  result=np.array(result)
  result=np.transpose(result)
  
  return result



'''实际三维点 转 像素坐标'''
def xyz2Color(x,y,z):
  #实际图像的相机内参数
  fx=921.305
  fy=921.583
  cx=648.475
  cy=359.169

  u=x*fx/z+cx
  v=y*fy/z+cy

  u=int(round(u))
  v=int(round(v))

  return u,v



'''模板点云 显示'''
def pcd_acquire(color_raw_name,depth_raw_name):

  color_raw=o3d.io.read_image(color_raw_name)
  depth_raw=o3d.io.read_image(depth_raw_name)
  depth_data = np.array(depth_raw)

  '''depth_scale为 缩放因子, 缩放为m, 该值与深度图的单位有关, 整型16位一般为mm, 所以需要缩放1000
     depth_trunc 为 截断距离 m
     convert_rgb_to_intensity 是否 转换为灰度图像'''
  rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000.0, depth_trunc=0.6, convert_rgb_to_intensity=False)

  '''创建模板点云显示的虚拟相机，注意要与采集相机相同'''
  inter=o3d.camera.PinholeCameraIntrinsic()
  inter.set_intrinsics(4096,3072,
                      1949.0157470703125,
                      1948.34521484375,
                      2053.27197265625,
                      1556.284912109375)

  #将模板点云转换到世界坐标系下（即转换到机器人的基坐标系下，以进行后续的识别与位姿估计）
  pcd=o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image,inter)

  return pcd,depth_data




#识别初始化
sift = cv.xfeatures2d.SIFT_create()

 #创建设置FLANN匹配，选择搜索算法
FLANN_INDEX_KDTREE = 0

index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)



'''实际 特征提取 与 点云获取'''
def feature_real_and_pcd(colorImg,depthImg): 

  '''实际 2D特征 与 描述符'''
  kp1, des1 = sift.detectAndCompute(colorImg, None)

  #注意在识别中需要保存为bgr图像，提高匹配准确度，但在显示过程中，需要将bgr转换为rgb进行点云显示
  color_rgb=cv.cvtColor(colorImg,cv.COLOR_BGR2RGB)

  '''显示 实际场景点云'''
  color_raw=o3d.geometry.Image(color_rgb)
  depth_raw=o3d.geometry.Image(depthImg)

  depth_data=np.array(depth_raw)

  rgbd_image=o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw, depth_scale=1000.0, depth_trunc=2, convert_rgb_to_intensity=False)

  inter=o3d.camera.PinholeCameraIntrinsic()
  inter.set_intrinsics(width_resolution,
                      height_resolution,
                      fx_camera,
                      fy_camera,
                      cx_camera,
                      cy_camera)

  pcd=o3d.geometry.PointCloud().create_from_rgbd_image(rgbd_image,inter)

  '''返回 实际特征描述符 实际2D特征 实际特征深度值 实际场景点云'''
  return des1,kp1,depth_data,pcd



'''模板 特征提取, 并与 实际 匹配'''
def feature_compare_match(imgName,viewAngle,des1,kp1,depth_data):
  img2 = cv.imread(imgName, 0)  #模板图像

  '''模板 2D特征 与 描述符
     返回值含义可参考 https://blog.csdn.net/qq_43279579/article/details/117687597'''
  kp2, des2 = sift.detectAndCompute(img2, None)

  '''匹配 两个 描述符集合，返回 与 每个实际特征点 描述符距离最近的 两个模板特征点
     返回值含义可参考 https://blog.csdn.net/weixin_44072651/article/details/89262277'''
  matches=flann.knnMatch(des1,des2,k=2)
  good=[]

  '''过滤 匹配度较高的 实际点 与 模板点'''
  for m,n in matches:
    if m.distance<match_ratio*n.distance:
      good.append(m)
      # print("m.queryIdx: {}, m.trainIdx: {}".format(m.queryIdx,m.trainIdx))
      # print("n.queryIdx: {}, n.trainIdx: {}".format(n.queryIdx,n.trainIdx))
      # print("--------------------------")

  '''判断过滤得到的点对 是否大于 最小设定的匹配点对数'''
  if len(good)>MIN_MATCH_COUNT:
      '''获取 实际点 的 像素坐标'''
      src_pts=np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
      '''获取 模板点 的 像素坐标'''
      dst_pts=np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

      dst_pts=np.squeeze(dst_pts)
      dst_pts=np.around(dst_pts)
      dst_pts=dst_pts.astype(int)

      src_pts=np.squeeze(src_pts)
      src_pts=np.around(src_pts)
      src_pts=src_pts.astype(int)


      # print("模板上的特征匹配点为：")
      # print(dst_pts)
      # print(dst_pts.shape)

      col=dst_pts[0,0]   #列 对应的是坐标x,对应的是像素u
      row=dst_pts[0,1]   #行 对应的是坐标y,对应的是像素v

      col2=dst_pts[2,0]   #列 对应的是坐标x,对应的是像素u
      row2=dst_pts[2,1]   #行 对应的是坐标y,对应的是像素v

  #深度值列表 空值化
  depth_value=[]
  #获取 模板特征点 对应的 深度值
  for i in range(len(dst_pts)):
    depth_value.append(depth_data[dst_pts[i,1],dst_pts[i,0]])

  '''将 模板2D点 转换为 3D点'''
  xyz_points=Depth2xyz(dst_pts[:,0],dst_pts[:,1],depth_value)

  TestFrame=[]
  # for i in range(len(dst_pts)):

  #     TestFrame_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(
  #         size=0.04, origin=xyz_points[i])

  #     # #即最终转换到了在观测程序obj_observe.py中的世界坐标系下
  #     # #先以世界坐标系进行旋转变换,注意True会导致显示的相机坐标存在网格误差
  #     # TestFrame_temp.rotate(np.linalg.inv(extrinsic)[0:3,0:3],False)
  #     # #再以世界坐标系进行平移变换，注意False会导致显示的相机坐标存在网格误差
  #     # TestFrame_temp.translate(np.linalg.inv(extrinsic)[0:3,3],True)

  #     TestFrame.append(TestFrame_temp)

  # for i in range(len(dst_pts)):
  #       vis.add_geometry(TestFrame[i])

  '''返回: 模板2D点 实际2D点 模板3D点 open3D的模板3D坐标系'''
  return dst_pts,src_pts,xyz_points,TestFrame



'''实际匹配点 转 3D点'''
def feature_real_match_show(src_pts,depth_data):
  #深度值列表 空值化
  depth_value=[]
  for i in range(len(src_pts)):
    #获取 实际特征点 对应的 深度值
    depth_value.append(depth_data[src_pts[i,1],src_pts[i,0]])

  '''将 实际2D点 转换为 3D点'''
  xyz_points=Depth2xyz2(src_pts[:,0],src_pts[:,1],depth_value)

  # TestFrame=[]
  # for i in range(len(src_pts)):
  #     # print(xyz_points[i])
  #     TestFrame_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(
  #         size=0.04, origin=xyz_points[i])
  #     TestFrame.append(TestFrame_temp)

  # for i in range(len(src_pts)):
  #     vis.add_geometry(TestFrame[i])

  '''返回 实际3D点'''
  return xyz_points



''' RANSAC-ICP'''
def RANSAC_sift(pcd_real,pcd_model,feature_points_real,feature_points_model,relative_Tran):
  #RANSAC算法利用较好的三维特征匹配点过滤模型
  '''设定 最大迭代次数 '''
  iters_max=5000

  '''当前最大迭代次数 初始化'''
  iters=iters_max
  
  '''位姿变换误差 初始化'''
  sigma=0.005

  '''最佳位姿变换 初始化'''
  best_T=[]
  best_R=[]
  best_t=[]

  '''上一次最佳内点数 初始化'''
  pretotal=0

  '''设定 求解概率'''
  P=0.9

  '''当前迭代次数'''
  i=0

  xyz_points2_origin=feature_points_real
  
  #实际场景点云坐标
  src_points=np.asarray(pcd_real.points)
  # dst_points=np.asarray(pcd_model.points)


  #定义 模板与实际 高配准3D点 的 坐标系集合
  high_mframe_set=[]
  high_rframe_set=[]


  #定义 模板与实际 高配准3D点 的 2D坐标点
  high_m2D_set=[]
  high_r2D_set=[]

  #定义 模板与实际 高配准3D点 集合
  high_matching_points_model=[]
  high_matching_points_real=[]


  while(i<iters):

    #获取 匹配的实际特征 长度
    siftmatch_points_length=feature_points_real.shape[0]

    #随机选择 一定数量的点，记录 编号
    sample_index=random.sample(range(siftmatch_points_length),model_solve_points)

    #获取 实际编号点 的 3D坐标
    xyz_points2_temp=feature_points_real[sample_index]
    #获取 模板编号点 的 3D坐标
    xyz_points_temp=feature_points_model[sample_index]

    #ICP求解 上述点对 的 变换位姿
    T, R, t = best_fit_transform(xyz_points2_temp, xyz_points_temp)

    #内点总数 初始化
    total_inlier=0

    #定义 模板 与 实际 高配准3D点 临时集合
    high_matching_points_model_temp=[]
    high_matching_points_real_temp=[]


   
    for index in range(len(feature_points_real)):
      #求 所有特征点 在求解变换下的 估计位姿
      xyz_point2_rotate_estimate=np.dot(R,feature_points_real[index].T)+t.T
      #求对应的 估计位姿 与 模板位姿 间的 误差 是否满足 精度阈值
      if(np.linalg.norm(xyz_point2_rotate_estimate-feature_points_model[index])<sigma):
        #内点 与 高配准 模板与实际点 累加
        total_inlier=total_inlier+1
        high_matching_points_model_temp.append(feature_points_model[index])
        high_matching_points_real_temp.append(feature_points_real[index])

    #判断 是否大于 上一次最佳内点数    
    if total_inlier>pretotal:
      #求 新的 最大迭代次数 并 赋值
      iters_temp=math.log(1-P)/math.log(1-pow(total_inlier/float(len(feature_points_real)),model_solve_points))
      if iters_temp<iters_max:
        iters=iters_temp

      '''保存 模板 与 实际 高配准3D点''' 
      high_matching_points_model=high_matching_points_model_temp
      high_matching_points_real=high_matching_points_real_temp

      #更新 上一次最佳内点数
      pretotal=total_inlier

      #保存为 最佳变换
      best_T=T
      best_R=R

    # print("-----------")
    # print("迭代次数"+str(i))
    # print("最大迭代次数"+str(iters))
    # print("内点数目"+str(pretotal))

    '''迭代次数累加'''
    i+=1
    #求取的 内点数目 大于2倍求解位姿模型所需点数目 与 所有匹配特征点数目的1/3 即可
    #因此若有 1/3的特征点 都能满足模型，从优化结果来看，模型大概率已经 足够精确，所以为了减少时耗，可 退出迭代
    if pretotal>=2*model_solve_points and pretotal>=0.8*len(feature_points_model):
      break


  #将 实际高配准3D点 变换为 2D坐标点 并保存
  for rpoint in high_matching_points_real:
    high_mu,high_mv=xyz2Color(rpoint[0],rpoint[1],rpoint[2])
    high_r2D_set.append([high_mu,high_mv])
    # cv.circle(circle_img, (high_mu,high_mv), 3, (255, 0, 0), 3)
  high_r2D_set=np.array(high_r2D_set)
  # high_r2Dmax_uv=np.max(high_r2D_set,axis=0)
  # high_r2Dmin_uv=np.min(high_r2D_set,axis=0)



  #将 模板高配准3D点 经 模板外参数变换 后 进行显示 
  '''即相当于 将模板目标坐标系 移动到与 相机坐标系 重合后， 得到重合后的 模板高配准点 坐标'''
  high_matching_points_model=np.array(high_matching_points_model)
  # mpoints_relative_Tran_inv=np.linalg.inv(relative_Tran)
  # mpoints_t=np.expand_dims(mpoints_relative_Tran_inv[0:3,3],0).repeat(high_matching_points_model.shape[0],axis=0)
  # mpoints_Tran=np.dot(mpoints_relative_Tran_inv[0:3,0:3],high_matching_points_model.T)+mpoints_t.T
  # mpoints_Tran=mpoints_Tran.T

  #创建 在Open3D中的 显示坐标系，并 保存
  # for frame_point in mpoints_Tran:
  #     high_mframe = o3d.geometry.TriangleMesh.create_coordinate_frame(
  #         size=0.02, origin=frame_point)
  #     high_mframe_set.append(high_mframe)

  # for i in range(len(mpoints_Tran)):
  #     vis.add_geometry(high_mframe_set[i])     



  '''将 实际场景点云 变换到 模板点云，也即 实际目标坐标系 变换到 模板坐标系，
     也即 相机定系下，模板坐标系 相对于 实际坐标系 的变换位姿，即为 左乘,即 结果 最后相对于 相机坐标系
     相对于 相机坐标系 进行变换(定系)：Tm=T变*Tr'''
  t=np.expand_dims(best_T[0:3,3],0).repeat(src_points.shape[0],axis=0)
  AA=np.dot(best_R,src_points.T)+t.T
  AA=AA.T
  pcd_real.points=o3d.utility.Vector3dVector(AA)
 
  '''求取最终的 实际目标坐标系 到 相机坐标系 的 位姿'''
  '''Tm=T变*Tr → Tr=T变^(-1)*Tm'''
  best_T_inv=np.linalg.inv(best_T)

  camera_to_real_Tran=np.dot(best_T_inv,relative_Tran)

  camera_to_real_Tran=np.dot(camera_to_real_Tran,model_bias_T)  # 模型中心高度设定

  # q=Quaternion(matrix=camera_to_real_Tran[0:3,0:3])
  # print(Quaternion(matrix=camera_to_real_Tran[0:3,0:3]))

  #转换为 四元数
  q2=SR.from_matrix(camera_to_real_Tran[0:3,0:3])
  camera_to_real_orientation=np.array(q2.as_quat())
  camera_to_real_position=np.array(camera_to_real_Tran[0:3,3])
  # print(camera_to_real_position)
  camera_to_real_pose=np.concatenate((camera_to_real_position,camera_to_real_orientation))
  


  '''将模板点云变换到实际点云，也即模板目标坐标系变换到实际坐标系重合，
     也即实际坐标系相对于模板坐标系的变换位姿，但为左乘，即结果相对于相机坐标系'''
  '''相对于相机坐标系进行变换(定系)：Tr=T变^(-1)*Tm'''
  # t=np.expand_dims(best_T_inv[0:3,3],0).repeat(dst_points.shape[0],axis=0)
  # BB=np.dot(best_T_inv[0:3,0:3],dst_points.T)+t.T
  # BB=BB.T
  # pcd_model.points=o3d.utility.Vector3dVector(BB)


  '''测试从 相机坐标系 到 匹配后的模板与实际点云坐标系（重合） 的 转换关系，但需将 实际坐标系 变换与 模板坐标系 重合
     由于在上一步中，将 实际场景点云 移动到与 模板点云重合，因此，此步中 可将 场景点云 与 模板 统一按照模板外参数 移动到 相机坐标系'''
  src_points_final=np.asarray(pcd_real.points)
  dst_points_final=np.asarray(pcd_model.points)

  relative_Tran_inv=np.linalg.inv(relative_Tran)
  # 实际场景点云 移动
  t=np.expand_dims(relative_Tran_inv[0:3,3],0).repeat(src_points_final.shape[0],axis=0)
  CC=np.dot(relative_Tran_inv[0:3,0:3],src_points_final.T)+t.T
  CC=CC.T
  pcd_real.points=o3d.utility.Vector3dVector(CC)

  # 模板点云 移动
  t=np.expand_dims(relative_Tran_inv[0:3,3],0).repeat(dst_points_final.shape[0],axis=0)
  DD=np.dot(relative_Tran_inv[0:3,0:3],dst_points_final.T)+t.T
  DD=DD.T
  pcd_model.points=o3d.utility.Vector3dVector(DD)

  
  '''测试从相机坐标系到匹配后的模板与实际点云坐标系的转换关系，但需将模板坐标系变换与实际坐标系重合'''
  # src_points_final=np.asarray(pcd_real.points)
  # dst_points_final=np.asarray(pcd_model.points)

  # camera_to_real_Tran_inv=np.linalg.inv(camera_to_real_Tran)
  # t=np.expand_dims(camera_to_real_Tran_inv[0:3,3],0).repeat(src_points_final.shape[0],axis=0)
  # CC=np.dot(camera_to_real_Tran_inv[0:3,0:3],src_points_final.T)+t.T
  # CC=CC.T
  # pcd_real.points=o3d.utility.Vector3dVector(CC)


  # t=np.expand_dims(camera_to_real_Tran_inv[0:3,3],0).repeat(dst_points_final.shape[0],axis=0)
  # DD=np.dot(camera_to_real_Tran_inv[0:3,0:3],dst_points_final.T)+t.T
  # DD=DD.T
  # pcd_model.points=o3d.utility.Vector3dVector(DD)

  '''返回 移动后的实际点云， 移动后的模板点云， 目标在相机坐标系下的四元数位姿，  目标在相机坐标系下的矩阵位姿，
          高配准的模板3D点集， 高配准的实际3D点集， 高配准的实际2D像素集合'''
  return pcd_real,pcd_model,camera_to_real_pose,camera_to_real_Tran,\
         high_matching_points_model,high_matching_points_real,high_r2D_set




'''获取 所选模板 的 外参数'''
def model_parameters_load(model_parameters_path,select):
  f=open(model_parameters_path, "r")

  lines = f.readlines()

  record_flag=0
  record_row=0

  record_parameters=[]

  for line in lines:
      if line.isspace():
          continue
      else:
          line_strip=line.strip()
          if line_strip==select:
            record_flag=1
            # print(line_strip)
          if record_flag==1:
            record_row=record_row+1
            print(record_row)
          if record_row>=2 and record_row<=5:
            record_parameters.append(line.strip())
          elif record_row>5:
            record_flag=0
            record_row=0

  # print(record_parameters)

  parameters_matrix_str=[]
  parameters_matrix=[]

  for parameter in record_parameters:
    parameters_matrix_str.append(re.findall(r"-?\d+\.?\d*",parameter))

  for parameters_str in parameters_matrix_str:
    parameters_matrix.append(list(map(float,parameters_str)))

  parameters_matrix=np.array(parameters_matrix)
  # print(parameters_matrix)

  parameters_Tran=np.zeros((4,4))
  parameters_Tran[0:3,3]=np.array(parameters_matrix[0,0:3]).T
  parameters_Tran[0:3,0:3]=np.array(parameters_matrix[1:4,0:3])
  parameters_Tran[3,3]=1.0

  return parameters_Tran



'''位姿估计'''
def realsense_pose_eatimation(camera_color_data,camera_depth_data):

    '''获得 模板目标 外参'''
    model_Tran=model_parameters_load(model2camera_parameters_path,model2camera_parameter_name)

    '''获得 模板点云 与 深度信息，需要全局路径参数'''
    pcd_model,depth_data_model=pcd_acquire(model_color_path,model_depth_path)

    
    '''获得 实际点云 与 实际2D特征点'''
    des1,kp1,depth_data_real,pcd_real=feature_real_and_pcd(camera_color_data,camera_depth_data)
    # des1,kp1,depth_data_real,pcd_real=feature_real_and_pcd(video_color,video_depth)

    ''' 模板与实际 2D匹配特征点 输出 与 模板3D匹配点'''
    dst_pts_model,src_pts_real,xyz_points_model,TestFrame_315=feature_compare_match(model_color_path,315,des1,kp1,depth_data_model)

    '''实际目标 3D特征点'''
    xyz_points_real=feature_real_match_show(src_pts_real,depth_data_real)

    '''相机坐标系下的 实际目标位姿，实际与模板点云, 实际与模板上的高配准3D点 实际对应的2D像素'''
    pcd_real_tran,pcd_model_tran,\
    camera_to_targt_pose,camera_to_targt_Tran,\
    high_mpoints,high_rpoints,high_target2D_set=RANSAC_sift(pcd_real,
                                                           pcd_model,
                                                           xyz_points_real,
                                                           xyz_points_model,
                                                           model_Tran)


    # print("camera_to_targt_pose")
    # print(camera_to_targt_pose)

    # #可视化
    # visualize(pcd_model_tran+pcd_real_tran)

    '''返回 相机坐标系下的实际目标位姿 实际目标上的2D像素'''
    return camera_to_targt_Tran,high_target2D_set

    # tcp_ip_server(camera_to_targt_pose)






class VideoPnP:
    def __init__(self):
      

      # #相机初始化
      self.video_fps=30
      self.video_w=1280
      self.video_h=720

      self.pipeline=rs.pipeline()
      self.config=rs.config()
      self.config.enable_stream(rs.stream.color,self.video_w,self.video_h,rs.format.bgr8,30)
      self.config.enable_stream(rs.stream.depth,self.video_w,self.video_h,rs.format.z16,30)
      self.pipeline.start(self.config)

      self.align_to = rs.stream.color
      self.align = rs.align(self.align_to)


      #彩色与深度图对齐,先放掉前边几帧,避免相机初始化造成图像质量波动
      for i in range(50):
        frames = self.pipeline.wait_for_frames()

      self.aligned_frames = self.align.process(frames)

      #获取对齐后的相机内参
      self.profile = self.aligned_frames.get_profile()
      self.intrinsics = self.profile.as_video_stream_profile().get_intrinsics()

      self.camera_matrix=np.array([[fx_camera,0,cx_camera],
                                   [0,fy_camera,cy_camera],
                                   [0,0,1]])


      #open3d初始化
      # 这里开始将realsense的数据转换为open3d的数据结构
      # 转换为open3d中的相机参数
      self.pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
                                      width_resolution, 
                                      height_resolution, 
                                      fx_camera, 
                                      fy_camera, 
                                      cx_camera, 
                                      cy_camera)


      # open3d 实时显示的初始化
      self.vis=o3d.visualization.Visualizer()
      '''创建相机初始坐标'''
      self.camera_coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
      '''创建目标初始坐标'''
      self.obj_coord=o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)


      #初始化 特征点数量最大值，需要根据 初始化的高配准特征点的数量 调整
      self.max_feature_number=200


      self.InitPose()

      self.InitParamSave()

      self.feature_points_show()



    '''初始化实际目标位姿  与 2D高配准特征框'''
    def InitPose(self):
      aligned_depth_frame = self.aligned_frames.get_depth_frame()
      color_frame = self.aligned_frames.get_color_frame()

      depth_image = np.asanyarray(aligned_depth_frame.get_data())
      color_image = np.asanyarray(color_frame.get_data())

      self.realsense_color_data=color_image
      self.realsense_depth_data=depth_image
      
      '''获取实际目标位姿  与 实际2D高配准特征点'''
      init_Tran,init_target2D_set=realsense_pose_eatimation(self.realsense_color_data,self.realsense_depth_data)

      '''获取 实际图像上高配准点的 最大与最小 像素坐标'''
      high_r2Dmax_uv=np.max(init_target2D_set,axis=0)
      high_r2Dmin_uv=np.min(init_target2D_set,axis=0)


      '''设定 2D追踪框 初始位置'''
      # 获取2D框 左上角点
      self.first_point = (high_r2Dmin_uv[0]-10, high_r2Dmin_uv[1]-10)
      # 获取2D框 右上角点
      self.last_point = (high_r2Dmax_uv[0]+10, high_r2Dmax_uv[1]+10)
      # 获取2D框 中心点
      self.src_pts_center=[(self.first_point[0]+self.last_point[0])/2,(self.first_point[1]+self.last_point[1])/2]
      self.src_pts_center=np.array(self.src_pts_center).astype(np.int)


      #目标位姿初始化
      rotation=init_Tran[0:3,0:3]

      rotation=np.array(rotation)
      translate=init_Tran[0:3,3]
      translate=np.array(translate)
      # print(rotation[0,:])
      self.obj_T=np.zeros((4,4))
      self.obj_T[0:3,0:3]=rotation
      self.obj_T[0:3,3]=translate
      self.obj_T[3,3]=1
      
      '''Open3D目标初始坐标'''
      self.obj_coord.rotate(self.obj_T[0:3,0:3],center=(0,0,0))
      self.obj_coord.translate(self.obj_T[0:3,3],True)



    '''保存初始化参数'''
    def InitParamSave(self):
      
      '''初始化 2D框选区域 彩色与深度图 全部为 空值'''
      self.roi_color=np.zeros((720,1280,3),dtype=np.uint8)
      self.roi_depth=np.zeros((720,1280),dtype=np.uint16)

      '''设置 2D框选区域 之外的 左上与右下截断点''' 
      first_cut_point=[max(self.first_point[0]-100,0),max(self.first_point[1]-100,0)]
      last_cut_point=[min(self.last_point[0]+100,1279),min(self.last_point[1]+100,719)]

      '''用截断点构成的 截断框 获取 目标彩色与深度纹理区域，作为 局部模板'''
      self.roi_color[first_cut_point[1]:last_cut_point[1],first_cut_point[0]:last_cut_point[0]]=\
      self.realsense_color_data[first_cut_point[1]:last_cut_point[1],first_cut_point[0]:last_cut_point[0]]
      
      self.roi_depth[first_cut_point[1]:last_cut_point[1],first_cut_point[0]:last_cut_point[0]]=\
      self.realsense_depth_data[first_cut_point[1]:last_cut_point[1],first_cut_point[0]:last_cut_point[0]]
      
      # print("self.roi_color.shape")
      # print(self.roi_color.shape)
      # print(self.roi_color.dtype)
      # print(self.realsense_color_data.dtype)
      # print(self.realsense_depth_data.dtype)
      # print(self.roi_depth[40,40])
      '''保存 该彩色与深度 局部区域'''
      cv.imwrite("test_param_color.png",self.roi_color)
      cv.imwrite("test_param_depth.png",self.roi_depth)

      self.init_obj_Tran=self.obj_T


    '''实际图像 深度 转 三维坐标'''
    def Depth2xyz2(self,u,v,depthValue):
      fx=fx_camera
      fy=fy_camera
      cx=cx_camera
      cy=cy_camera
      
      depthValue=np.array(depthValue)
      u=np.array(u)
      v=np.array(v)

      z=depthValue*0.001
      x=((u-cx)*z)/fx
      y=((v-cy)*z)/fy

      result=[x,y,z]
      result=np.array(result)
      result=np.transpose(result)
      
      return result



    '''深度值获取'''
    def DistanceAcquire(self,pts,depth_image):
       
      distances=[]
      # print(pts.shape)
      for point_2d in pts:
          # print(point_2d)
          #判断是否 超出 图像区域，若超出，则按照顶点的深度值 设置 该点像素深度值
          if point_2d[0]>1279 or point_2d[1]>719:
            distances.append([depth_image[719,1279]])
          else:  
            distances.append([depth_image[point_2d[1],point_2d[0]]])

      distances=np.array(distances)      
      # pnp_pts=np.array(pnp_pts_temp)

      return distances



    '''特征点显示'''
    def feature_points_show(self):

      #创建实时显示窗口
      self.vis.create_window()

      #为 特征点 分配足够多的 open3d坐标显示
      for ff in range(self.max_feature_number):
          # 动态变量执行
          exec('TestFrame_temp{} = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0,0,0])'.format(ff))
          exec('self.vis.add_geometry(TestFrame_temp{})'.format(ff))

      # open3D实时显示 初始化
      self.pcd_new = o3d.geometry.PointCloud()
      #注意实时显示要进行如下初始化点云，否则无法显示
      points = [[-0.5,-0.5,-0.5],[0.5,0.5,0.5]]
      # colors = [[-0.5,-0.5,-0.5],[0.5,0.5,0.5]]
      self.pcd_new.points= o3d.utility.Vector3dVector(points)
      ''' 将 实时点云 相机坐标系 目标坐标系 加入到 open3d中'''
      self.vis.add_geometry(self.camera_coord)
      self.vis.add_geometry(self.obj_coord)
      self.vis.add_geometry(self.pcd_new)

      # #控制视场角
      # ctr = self.vis.get_view_control()
      # camera_parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2023-09-10-11-54-13.json")
      # ctr.convert_from_pinhole_camera_parameters(camera_parameters)

      '''更新并显示'''
      self.vis.poll_events()
      self.vis.update_renderer()


      #记录帧数
      ii=0

      # ShiTomasi corner detection的参数 创建字典类型
      self.feature_params = dict(maxCorners=self.max_feature_number,
                            qualityLevel=0.01,
                            minDistance=15,
                            blockSize=5)
      # 光流法参数 创建字典类型
      self.lk_params = dict(winSize=(17, 17),
                      maxLevel=2,
                      criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

      # 创建随机生成的颜色 500个
      self.color = np.random.randint(0, 255, (500, 3))

      '''创建与实时画面 相同大小的 待截断灰度图'''
      old_gray_cut=np.zeros((self.video_h,self.video_w))
      old_gray_cut=old_gray_cut.astype(np.uint8)

      '''取出视频的RGB图 与gepth图 第一帧 '''
      old_frame = self.realsense_color_data                             
      depth_image=self.realsense_depth_data                             
      ii+=1  #注意由于彩色图与深度图帧是一一对应的，因此当彩色图读取一帧时，为了记录当前帧的索引，需要记录,以下同理

      # 灰度化
      old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)  
      # print(old_gray.dtype)
      # print(old_gray_cut.dtype)
      '''使用 2D追踪框 来获取 截断灰度图'''
      old_gray_cut[self.first_point[1]:self.last_point[1],self.first_point[0]:self.last_point[0]]=\
      old_gray[self.first_point[1]:self.last_point[1],self.first_point[0]:self.last_point[0]]
      # cv.imshow('test', old_gray_cut)
      # cv.waitKey()
      
      '''获取 第一帧截断灰度图 特征点'''
      p0 = cv.goodFeaturesToTrack(old_gray_cut, mask=None, **self.feature_params)
      mask = np.zeros_like(old_frame)                         # 为绘制创建掩码图片
      
      '''获取第一帧截断灰度图 特征点 的深度值'''
      first_pts=np.squeeze(p0).astype(np.int)
      distances0=self.DistanceAcquire(first_pts,depth_image)
      

      while True:
          #计时准备
          self.t1=time.time()
          
          '''获取 实时图像'''
          frames = self.pipeline.wait_for_frames()
          aligned_frames = self.align.process(frames)

          aligned_depth_frame = aligned_frames.get_depth_frame()
          color_frame = aligned_frames.get_color_frame()

          self.depth_image = np.asanyarray(aligned_depth_frame.get_data())
          self.color_image = np.asanyarray(color_frame.get_data())

          ii+=1   #查看上述说明，记录帧数

          '''获取 当前实时图像 的 灰度图'''
          frame_gray = cv.cvtColor(self.color_image, cv.COLOR_BGR2GRAY)
          # print(type(p0))

          '''判断上一帧的特征点（待计算特征点）是否在追踪框内，注意需要同步求取对应的深度值'''
          p_temp=[]
          p_depth_temp=[]
          # print(len(p0))
          # print(distances0.shape)
          for i_pt in range(len(p0)):
             if (p0[i_pt][0][0]>self.first_point[0]) and (p0[i_pt][0][0]<self.last_point[0]):
                if (p0[i_pt][0][1]>self.first_point[1]) and (p0[i_pt][0][1]<self.last_point[1]):
                   p_temp.append(p0[i_pt])
                   p_depth_temp.append(distances0[i_pt])
          p_temp=np.array(p_temp)
          p_depth_temp=np.array(p_depth_temp)
          
          '''更新 待计算的特征点 及其 深度值，即更新 上一帧'''
          p0=p_temp
          distances0=p_depth_temp
          # print(distances0.shape)
          # print(type(p_temp))

          try:
            '''计算 上一帧与当前帧之间的光流 以获取 特征点新位置，即作为 当前帧上的对应特征点'''
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **self.lk_params)
  
            # print(p0[10])
            # print(p1[10])         
            #获取当前帧的特征点对应的深度值
            current_pts=np.squeeze(p1).astype(np.int)
            # print(np.max(current_pts,axis=0))
            distances1=self.DistanceAcquire(current_pts,self.depth_image)

            # print(st)
            # print(type(st))
            '''选择具有光流追踪的 当前帧 与 上一帧 的特征点 及其深度值'''
            self.good_new = p1[st == 1]
            # print(p1.shape)
            # print(self.good_new.shape)
            self.good_new_distance=distances1[st == 1]

            self.good_old = p0[st == 1]
            self.good_old_distance=distances0[st == 1]
          except:
            pass

          '''初始化用以epnp算法求解 当前位姿的 当前帧与上一帧的 2D光流点及其深度值'''
          epnp_distance0=[]
          epnp_distance1=[]
          epnp_pts0=[]
          epnp_pts1=[]
          # print(type(epnp_pts0))
          
          for epnp_i in range(len(self.good_old)):
             #判断 当前帧与上一帧 较好的光流点 是否具有 深度值
             if self.good_old_distance[epnp_i]!=0:
                 if self.good_new_distance[epnp_i]!=0:
                    epnp_pts0.append(self.good_old[epnp_i])
                    epnp_pts1.append(self.good_new[epnp_i])
                    epnp_distance0.append(self.good_old_distance[epnp_i])
                    epnp_distance1.append(self.good_new_distance[epnp_i])

          epnp_pts0=np.array(epnp_pts0)
          epnp_pts1=np.array(epnp_pts1)
          epnp_distance0=np.array(epnp_distance0)
          epnp_distance1=np.array(epnp_distance1)          

          '''将 用以求解实时位姿的 过滤下来的 当前帧与上一帧光流点 转换为 3D点 '''
          try:
            points_3d0=self.Depth2xyz2(epnp_pts0[:,0],epnp_pts0[:,1],epnp_distance0)
            points_3d1=self.Depth2xyz2(epnp_pts1[:,0],epnp_pts1[:,1],epnp_distance1)
          except:
            pass
          
          # print(epnp_pts0.shape)
          # print(epnp_pts1.shape)
          # print(epnp_distance0.shape)
          # print(epnp_distance1.shape)

          # (success, rotation_vector, translation_vector) = cv2.solvePnP(points_3d0.take([1,5,10,20]), 
          #                                                               epnp_pts1.take([1,5,10,20]), 
          #                                                               self.camera_matrix, 
          #                                                               distCoeffs=None,
          #                                                               flags=cv2.SOLVEPNP_EPNP)
          '''求解 当前帧的位姿'''
          try:
            success, rotation_vector, translation_vector, inliers=cv.solvePnPRansac(points_3d0,
                                                                                   epnp_pts1,
                                                                                   self.camera_matrix,
                                                                                   distCoeffs=None,
                                                                                   iterationsCount=1000,
                                                                                   flags=cv.SOLVEPNP_P3P)
          except:
             pass
          # print(rotation_vector)
          # print(epnp_pts1[10])
          # print(epnp_pts0[10])
          # print(translation_vector)
          # print(cv2.Rodrigues(rotation_vector)[0])

          #转换为齐次变换矩阵
          self.Transport=np.zeros((4,4))
          self.Transport[0:3,0:3]=cv.Rodrigues(rotation_vector)[0]
          self.Transport[0,3]=translation_vector[0]
          self.Transport[1,3]=translation_vector[1]
          self.Transport[2,3]=translation_vector[2]
          self.Transport[3,3]=1

          # print(self.Transport)

          self.Transport_inv=np.linalg.inv(self.Transport)

          '''判断求解的 移动向量 是否在 阈值内
             如果移动量较小，说明符合连续规律，如果移动量较大，说明可能出现求解错误'''
          if (self.Transport[0,3]<=0.03) and  (self.Transport[0,3]>=-0.03):
            self.obj_T=np.dot(self.Transport,self.obj_T)

          # else:
            #  print("self.Transport")
            #  print(self.Transport)
            #  print(points_3d0.shape)
            #  print(epnp_pts1.shape)


          # print(self.Transport_inv[0:3,3])
          # print(self.obj_T)
          # if self.obj_T[0,3]<-1.5:
          #    print("self.Transport")
          #    print(self.Transport)
          
          

          obj_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0,0,0])


          # obj_temp.rotate(self.obj_T[0:3,0:3],False)
          # obj_temp.translate(self.obj_T[0:3,3],True)


          self.refin_fixtime_obj()



          obj_temp.rotate(self.camera_to_real_Tran[0:3,0:3],center=(0,0,0))
          obj_temp.translate(self.camera_to_real_Tran[0:3,3],True)



          self.obj_coord.vertices=obj_temp.vertices

          # self.obj_coord.rotate(np.array(cv2.Rodrigues(rotation_vector)[0]),False)


          print("===================")

          # 绘制跟踪框
          for i, (new, old) in enumerate(zip(self.good_new, self.good_old)):
              a, b = new.ravel()  #将数组拉为一维
              c, d = old.ravel()
              mask = cv.line(mask, (int(a), int(b)), (int(c),int(d)), self.color[i].tolist(), 2)
              frame = cv.circle(self.color_image, (int(a),int(b)), 5, self.color[i].tolist(), -1)
          img = cv.add(frame, mask)
          print("self.first_point: {}".format(self.first_point))
          print("self.last_point: {}".format(self.last_point))
          cv.rectangle(self.color_image, self.first_point, self.last_point, (0, 255, 0), 2)
          cv.imshow('frame', self.color_image)
          k = cv.waitKey(30)  # & 0xff
          if k == 27:
              break
          old_gray = frame_gray.copy()

          #将 当前帧 的具有光流追踪的特征点 更新赋值给 上一帧,注意为了采用条件引用的方式，需要按照st的维度添加一个维度
          p0 = self.good_new.reshape(-1, 1, 2)
          distances0=self.good_new_distance.reshape(-1,1)
          # print(distances0.shape)

          #求取下一帧追踪框的中心点
          src_pts=np.squeeze(p0)      
          self.src_pts_center=np.mean(src_pts,axis=0).astype(np.int)
          # print(self.src_pts_center)

          try:
            self.first_point=(self.src_pts_center[0]-100,self.src_pts_center[1]-200)
            self.last_point=(self.src_pts_center[0]+100,self.src_pts_center[1]+200)
          # print(first_point)
          # print(last_point)
          except:
            pass

          #取出当前帧 的具有光流追踪的特征点          
          # pnp_pts=src_pts.astype(np.int)
          # print(src_pts.shape)




          # distances=[]
          # pnp_pts_temp=[]
          # for point_2d in pnp_pts:
          #   if depth_image[point_2d[1],point_2d[0]] !=0:
          #      pnp_pts_temp.append(point_2d)
          #      distances.append(depth_image[point_2d[1],point_2d[0]])
               
          # pnp_pts=np.array(pnp_pts_temp)


          
          # print(src_pts.shape)
          # print(len(distances))
          # print("==============")
          # print(distances)
          # print(src_pts)

          # points_3d=self.Depth2xyz2(pnp_pts[:,0],pnp_pts[:,1],distances)
          # print(points_3d.shape)


          # for kk in range(len(points_3d0)):
          #   frame_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=points_3d0[kk])
          #   exec('TestFrame_temp{}.vertices=frame_temp.vertices'.format(kk))   #动态变量执行

          # for gg in range(len(points_3d0),200):
          #   frame_temp = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0,0,0])
          #   exec('TestFrame_temp{}.vertices=frame_temp.vertices'.format(gg))   #动态变量执行
          

          self.color_image=cv.cvtColor(self.color_image, cv.COLOR_RGB2BGR)
          

          img_depth = o3d.geometry.Image(self.depth_image)
          img_color = o3d.geometry.Image(self.color_image)

          #从彩色图、深度图创建RGBD
          rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(img_color, img_depth,convert_rgb_to_intensity=False)
          # 创建pcd
          pcd_temp = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,self.pinhole_camera_intrinsic)

          # pcd_temp.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


          self.pcd_new.points = pcd_temp.points
          self.pcd_new.colors=pcd_temp.colors

          if self.total_features<11:
            print(self.total_features)
            self.obj_coord_zero = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
            self.obj_coord.vertices=self.obj_coord_zero.vertices


          self.vis.update_geometry(self.camera_coord)
          self.vis.update_geometry(self.obj_coord)
          self.vis.update_geometry(self.pcd_new)

          # for jj in range(200):
          #     # print(xyz_points[i])
          #     exec('self.vis.update_geometry(TestFrame_temp{})'.format(jj)) 
          # vis.add_geometry(pcd_new)
          self.vis.poll_events()
          self.vis.update_renderer()


          self.t2=time.time()
          # print("t2-t1:{}".format(self.t2-self.t1))




    '''实时实际 特征提取 与 点云获取'''
    def local_feature_real_and_pcd(self):
      # img1=cv.imread("target_color3.png",0)
      # img1=cv.imread(colorImgName,0)
      # print("img1.shape")
      # print(img1.shape)   

      '''实际 2D特征 与 描述符'''
      self.kp1, self.des1 = sift.detectAndCompute(self.color_image, None)

      #注意在识别中需要保存为bgr图像，提高匹配准确度，但在显示过程中，需要将bgr转换为rgb进行点云显示
      color_rgb=cv.cvtColor(self.color_image,cv.COLOR_BGR2RGB)
      color_raw=o3d.geometry.Image(color_rgb)
      depth_raw=o3d.geometry.Image(self.depth_image)

      depth_data=np.array(depth_raw)
      
    
    

    '''局部模板 特征提取, 并与 实际 匹配'''
    def local_feature_compare_match(self):
      # img2 = cv.imread(self.realsense_color_data, 0)  #模板图像

      img2=cv.cvtColor(self.roi_color,cv.COLOR_BGR2GRAY)

      #求 实际特征点 与 描述符
      self.kp2, self.des2 = sift.detectAndCompute(img2, None)

      '''匹配 两个 描述符集合，返回 与 每个实际特征点 描述符距离最近的 两个模板特征点
      返回值含义可参考 https://blog.csdn.net/weixin_44072651/article/details/89262277'''
      matches=flann.knnMatch(self.des1,self.des2,k=2)
      good=[]

      '''过滤 匹配度较高的 实际点 与 模板点'''
      for m,n in matches:
        if m.distance<0.7*n.distance:
          good.append(m)
      # print(len(good))
      '''判断过滤得到的点对 是否大于 最小设定的匹配点对数'''
      if len(good)>MIN_MATCH_COUNT:
          src_pts=np.float32([self.kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
          dst_pts=np.float32([self.kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

          dst_pts=np.squeeze(dst_pts)
          dst_pts=np.around(dst_pts)
          self.dst_pts=dst_pts.astype(int)

          src_pts=np.squeeze(src_pts)
          src_pts=np.around(src_pts)
          self.src_pts=src_pts.astype(int)

          # print("模板上的特征匹配点为：")
          # print(dst_pts)
          # print(dst_pts.shape)

          col=self.dst_pts[0,0]   #列 对应的是坐标x,对应的是像素u
          row=self.dst_pts[0,1]   #行 对应的是坐标y,对应的是像素v

          col2=self.dst_pts[2,0]   #列 对应的是坐标x,对应的是像素u
          row2=self.dst_pts[2,1]   #行 对应的是坐标y,对应的是像素v

      #深度值列表 空值化
      depth_value=[]
      #获取 局部模板特征点 对应的 深度值
      for i in range(len(self.dst_pts)):
        depth_value.append(self.roi_depth[self.dst_pts[i,1],self.dst_pts[i,0]])
      '''将 局部模板2D点 转换为 3D点'''
      self.xyz_points_init=Depth2xyz2(self.dst_pts[:,0],self.dst_pts[:,1],depth_value)



    '''实时 实际匹配点 转 3D点'''
    def local_feature_real_match_show(self):
      depth_value=[]
      for i in range(len(self.src_pts)):
        depth_value.append(self.depth_image[self.src_pts[i,1],self.src_pts[i,0]])
      '''获取实时 实际3D特征点'''
      self.xyz_points_realtime=Depth2xyz2(self.src_pts[:,0],self.src_pts[:,1],depth_value)




    '''实时 局部 RANSAC-ICP'''
    def local_RANSAC_sift(self):
      #RANSAC算法利用较好的三维特征匹配点过滤模型
      iters_max=1000
      iters=iters_max
      sigma=0.002


      best_T=[]
      best_R=[]
      best_t=[]
      pretotal=0
      P=0.9
      i=0


      while(i<iters):

        #获取 匹配的实时实际特征 长度
        siftmatch_points_length=self.xyz_points_realtime.shape[0]

        #随机选择 一定数量的点，记录 编号
        sample_index=random.sample(range(siftmatch_points_length),model_solve_points)

        #获取 实际编号点 的 3D坐标
        xyz_points2_temp=self.xyz_points_realtime[sample_index]
        #获取 模板编号点 的 3D坐标
        xyz_points_temp=self.xyz_points_init[sample_index]

        #ICP求解 上述点对 的 变换位姿
        T, R, t = best_fit_transform(xyz_points2_temp, xyz_points_temp)

        #内点总数 初始化
        total_inlier=0

        #定义 模板 与 实际 高配准3D点 临时集合
        high_matching_points_model_temp=[]
        high_matching_points_real_temp=[]


        for index in range(len(self.xyz_points_realtime)):
          #求 所有特征点 在求解变换下的 估计位姿
          xyz_point2_rotate_estimate=np.dot(R,self.xyz_points_realtime[index].T)+t.T
          #求对应的 估计位姿 与 模板位姿 间的 误差 是否满足 精度阈值
          if(np.linalg.norm(xyz_point2_rotate_estimate-self.xyz_points_init[index])<sigma):
            #内点 与 高配准 模板与实际点 累加
            total_inlier=total_inlier+1
            high_matching_points_model_temp.append(self.xyz_points_init[index])
            high_matching_points_real_temp.append(self.xyz_points_realtime[index])

        #判断 是否大于 上一次最佳内点数 
        if total_inlier>pretotal:
          #判断 当前内点总数 是否等于 特征点总数
          if(total_inlier==len(self.xyz_points_realtime)):
            #若等于 则需要 对分母添加 修正因子，求 当前最大迭代次数
            iters_temp=math.log(1-P)/math.log(1.0-pow(total_inlier/float(len(self.xyz_points_realtime)+1),model_solve_points))
          else:
            #若不等于 则按照 正常RANSAC算法 求 当前最大迭代次数
            iters_temp=math.log(1-P)/math.log(1.0-pow(total_inlier/float(len(self.xyz_points_realtime)),model_solve_points))
          #若小于 更新 最大迭代次数
          if iters_temp<iters_max:
            iters=iters_temp

          #更新 上一次最佳内点数， 并保存 最佳变换
          pretotal=total_inlier
          best_T=T
          best_R=R

        '''迭代次数累加'''
        i+=1
        #求取的内点数目 大于2倍求解位姿模型所需点数目 与 所有匹配特征点数目的1/3即可
        #因此若有1/3的特征点都能满足模型，从优化结果来看，模型大概率已经足够精确，所以为了减少时耗，可退出迭代
        if pretotal>=2*model_solve_points and pretotal>=0.5*len(self.xyz_points_init):
          break

      '''将实际点云变换到模板点云，也即实际目标坐标系变换到模板坐标系，
        也即相机定系下，模板坐标系相对于实际坐标系的变换位姿，即为左乘,即结果最后相对于相机坐标系'''
      '''相对于相机坐标系进行变换(定系)：Tm=T变*Tr'''
      # t=np.expand_dims(best_T[0:3,3],0).repeat(src_points.shape[0],axis=0)
      # AA=np.dot(best_R,src_points.T)+t.T
      # AA=AA.T
      # pcd_real.points=o3d.utility.Vector3dVector(AA)

      '''求取最终的实际目标坐标系到相机坐标系的位姿'''
      '''Tm=T变*Tr → Tr=T变^(-1)*Tm'''
      try:
        self.best_T_inv=np.linalg.inv(best_T)
      except:
        pass

      self.camera_to_real_Tran=np.dot(self.best_T_inv,self.init_obj_Tran)
    


    def refin_fixtime_obj(self):

        '''获得实际局部点云与局部2D特征点'''
        self.local_feature_real_and_pcd()

        '''模板与实际的2D匹配特征点输出 与 模板的3D匹配点输出'''
        self.local_feature_compare_match()

        '''直方图统计 判断 特征是否集中，以确定是否可能为 目标'''
        self.picture_points_show()

        '''判断主要区域的特征数 是否 满足求解要求'''
        if self.total_features<4*model_solve_points:
          return

        '''实际目标的3D特征点'''
        self.local_feature_real_match_show()

        '''相机坐标系变换到实际目标的位姿，即实际目标相对于相机的变换位姿，以及点云输出,实际目标上的高配准点'''
        self.local_RANSAC_sift()
        # print("camera_to_targt_pose")
        # print()

      # #可视化
      # visualize(pcd_model_tran+pcd_real_tran)

    '''统计 第一与第二特征区域 的 总特征点数'''
    def picture_points_show(self):
      if len(self.src_pts)!=0:
        # print("self.src_pts")
        '''直方图统计'''
        n, bins, patches=plt.hist(self.src_pts[:,0],bins=40,range=(0,1280))
        # plt.ion()
        # plt.pause(2)
        # plt.clf()
        # plt.close()
        # print("n")
        # print(n)
        '''根据直方图统计确定最多特征区域 的 特征点数'''
        total_features=np.max(n)
        # print(np.argmax(n))
        '''去除最大特征区域'''
        new_n=np.delete(n,np.argmax(n))
        # print(new_n)
        '''判断剩下的 最多特征区域 的 特征点数，并与最多特征区域的特征点数合并'''
        self.total_features=total_features+np.max(new_n)
        # print(total_features)
        # print("bins")
        # print(bins)
        # print(bins[np.argmax(n)+1])
        # print("patches")
        # print(patches)          






if __name__=="__main__":
  VideoPnP()