import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr, read_matrix
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture


def fuse_me(world_pos, world_pos_1, mv, mv_1, color, depth, stencil, vp_matrix, vp_matrix_next, program):
    height, width = color.shape[0], color.shape[1]

    # 使用传入的ndarray中的数据创建纹理
    world_pos_tex = create_texture(world_pos, width, height)
    world_pos_1_tex = create_texture(world_pos_1, width, height)
    mv_tex = create_texture(mv, width, height)
    mv_1_tex = create_texture(mv_1, width, height)
    depth_tex = create_texture(depth, width, height)
    color_tex = create_texture(color, width, height)
    stencil_tex = create_texture(stencil, width, height)

    # 创建存放结果的纹理
    warp_color_tex = create_texture(None, width, height)
    warp_mv_tex = create_texture(None, width, height)
    warp_depth_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)

    # 将纹理绑定到着色器
    in_textures = [world_pos_tex, world_pos_1_tex, mv_tex, mv_1_tex, depth_tex, color_tex, stencil_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [warp_color_tex, warp_mv_tex, warp_depth_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    # 绑定uniform矩阵
    vp_loc = glGetUniformLocation(program, "vp_matrix")
    if vp_loc >= 0:
        glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix)
    vp_loc = glGetUniformLocation(program, "vp_matrix_next")
    if vp_loc >= 0:
        glUniformMatrix4fv(vp_loc, 1, GL_TRUE, vp_matrix_next)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)
    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # 从结果纹理中读取数据
    warp_color = read_texture(warp_color_tex, width, height)
    warp_mv = read_texture(warp_mv_tex, width, height)
    warp_depth = read_texture(warp_depth_tex, width, height, GL_RED_INTEGER, GL_UNSIGNED_INT)
    warp_depth = ((2147483647 - np.expand_dims(warp_depth, axis=-1)) / 65535).astype(np.float32)
    warp_color = np.reshape(warp_color, (height, width, 4))
    warp_mv = np.reshape(warp_mv, (height, width, 4))
    warp_depth = np.reshape(warp_depth, (height, width, 1))

    glDeleteTextures(10, [world_pos_tex, world_pos_1_tex, mv_tex, mv_1_tex, depth_tex, color_tex, stencil_tex, warp_color_tex, warp_mv_tex, warp_depth_tex])
    glFinish()

    return warp_color, warp_mv, warp_depth


def fuse_me_init():
    # 初始化opengl和创建着色器
    create_window(1280, 720, "fuse_me")
    with open("shaders/fuse_me.comp", "r") as f:
        shader_source = f.read()
    shader = create_compute_shader(shader_source)
    program = create_compute_program(shader)
    glUseProgram(program)
    return program


def fuse_me_main(label_index, label_path, seq_path, save_path, scene_name, program, debug=False):
    """
    :param label_index: 需要预测的标签帧的索引
    :param label_path: 标签帧的路径
    :param seq_path: 序列帧的路径
    :param save_path: 保存路径
    :param scene_name: 场景名称
    :param program: 着色器程序
    """
    input_index = (label_index - 1) // 2
    world_pos = read_exr(os.path.join(label_path, f"{scene_name}WorldPosition.{str(label_index-1).zfill(4)}.exr"), channel=4)
    world_pos_1 = read_exr(os.path.join(label_path, f"{scene_name}WorldPosition.{str(label_index-3).zfill(4)}.exr"), channel=4)
    mv = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index).zfill(4)}.exr"), channel=4)
    mv_1 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index-1).zfill(4)}.exr"), channel=4)
    depth = read_exr(os.path.join(label_path, f"{scene_name}WorldNormalAndSceneDepth.{str(label_index-1).zfill(4)}.exr"), channel=4)[:, :, 3:4]
    depth = np.repeat(depth, 4, axis=-1)
    color = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-1).zfill(4)}.exr"), channel=4)
    stencil = read_exr(os.path.join(label_path, f"{scene_name}MyStencil.{str(label_index-1).zfill(4)}.exr"), channel=4)
    
    vp_matrix = read_matrix(os.path.join(label_path, f"{scene_name}Matrix.{str(label_index-1).zfill(4)}.txt"))
    vp_matrix_next = read_matrix(os.path.join(label_path, f"{scene_name}Matrix.{str(label_index).zfill(4)}.txt"))

    warp_color, warp_mv, warp_depth = fuse_me(world_pos, world_pos_1, mv, mv_1, depth, color, stencil, vp_matrix, vp_matrix_next, program)
    
    write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), warp_color)
    write_exr(os.path.join(save_path, f"{scene_name}WarpMotionVector.{str(label_index).zfill(4)}.exr"), warp_mv)

    if debug:
        # 读取并保存label
        color_gt = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), channel=4)
        mv_gt = read_exr(os.path.join(label_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), channel=4)
        write_exr(os.path.join(save_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), color_gt)
        write_exr(os.path.join(save_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), mv_gt)