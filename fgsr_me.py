import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture


def fgsr_me(mv, mv_1, depth, color):
    height, width = color.shape[0], color.shape[1]
    # mv = np.reshape(mv, (width, height, 4))
    # mv_1 = np.reshape(mv_1, (width, height, 4))
    # depth = np.reshape(depth, (width, height, 4))
    # color = np.reshape(color, (width, height, 4))

    # 使用传入的ndarray中的数据创建纹理
    mv_tex = create_texture(mv, width, height)
    mv_1_tex = create_texture(mv_1, width, height)
    depth_tex = create_texture(depth, width, height)
    color_tex = create_texture(color, width, height)

    # 创建存放结果的纹理
    warp_color_tex = create_texture(None, width, height)
    warp_mv_tex = create_texture(None, width, height)
    warp_depth_tex = create_texture(None, width, height, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT)

    # 将纹理绑定到着色器
    in_textures = [mv_tex, mv_1_tex, depth_tex, color_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [warp_color_tex, warp_mv_tex, warp_depth_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

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

    delete_textures = [mv_tex, mv_1_tex, depth_tex, color_tex, warp_color_tex, warp_mv_tex, warp_depth_tex]
    glDeleteTextures(len(delete_textures), delete_textures)
    glFinish()

    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL Error: {error}")

    return warp_color, warp_mv, warp_depth


def fgsr_me_init():
    # 初始化opengl和创建着色器
    create_window(1280, 720, "fgsr_me")
    with open("shaders/fgsr_me.comp", "r") as f:
        shader_source = f.read()
    shader = create_compute_shader(shader_source)
    program = create_compute_program(shader)
    glUseProgram(program)
    return program


def fgsr_me_main(label_index, label_path, seq_path, save_path, scene_name, program, debug=False):
    input_index = (label_index - 1) // 2
    mv = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index).zfill(4)}.exr"), channel=4)
    mv_1 = read_exr(os.path.join(seq_path, f"{scene_name}MotionVector.{str(input_index-1).zfill(4)}.exr"), channel=4)
    depth = read_exr(os.path.join(label_path, f"{scene_name}SceneDepthAndNoV.{str(label_index-1).zfill(4)}.exr"), channel=4)[:, :, 2:3]
    color = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index-1).zfill(4)}.exr"), channel=4)
    depth = np.repeat(depth, 4, axis=-1)
    
    warp_color, warp_mv, warp_depth = fgsr_me(mv, mv_1, depth, color)
    
    write_exr(os.path.join(save_path, f"{scene_name}WarpColor.{str(label_index).zfill(4)}.exr"), warp_color)
    write_exr(os.path.join(save_path, f"{scene_name}WarpMotionVector.{str(label_index).zfill(4)}.exr"), warp_mv)
    # write_exr(os.path.join(save_path, f"{scene_name}WarpDepth.{str(i+1).zfill(4)}.exr"), warp_depth)

    if debug:
        # 读取并保存label
        color_gt = read_exr(os.path.join(label_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), channel=4)
        mv_gt = read_exr(os.path.join(label_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), channel=4)
        write_exr(os.path.join(save_path, f"{scene_name}PreTonemapHDRColor.{str(label_index).zfill(4)}.exr"), color_gt)
        write_exr(os.path.join(save_path, f"{scene_name}MotionVector.{str(label_index).zfill(4)}.exr"), mv_gt)
