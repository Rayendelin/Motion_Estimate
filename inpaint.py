import os
import numpy as np
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from data_io import read_exr, write_exr, read_matrix
from opengl_util import create_compute_shader, create_compute_program, create_texture, create_window, read_texture


def inpaint(mv):
    height, width = mv.shape[0], mv.shape[1]

    # 使用传入的ndarray中的数据创建纹理
    mv_tex = create_texture(mv, width, height)

    # 创建存放结果的纹理
    inpaint_mv_tex = create_texture(None, width, height)

    # 将纹理绑定到着色器
    in_textures = [mv_tex]
    glBindTextures(0, len(in_textures), in_textures)
    out_textures = [inpaint_mv_tex]
    glBindImageTextures(0, len(out_textures), out_textures)

    num_groups_x = math.ceil(width / 8)
    num_groups_y = math.ceil(height / 8)
    glDispatchCompute(num_groups_x, num_groups_y, 1)
    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    # 从结果纹理中读取数据
    inpaint_mv = read_texture(inpaint_mv_tex, width, height)
    inpaint_mv = np.reshape(inpaint_mv, (height, width, 4))

    glDeleteTextures(2, [mv_tex, inpaint_mv_tex])
    glFinish()

    return inpaint_mv


def inpaint_init():
    # 初始化opengl和创建着色器
    create_window(1280, 720, "inpaint")
    with open("shaders/fgsr_inpaint.comp", "r") as f:
        shader_source = f.read()
    shader = create_compute_shader(shader_source)
    program = create_compute_program(shader)
    glUseProgram(program)
    return program


def inpaint_main(label_index, label_path, save_path, scene_name):
    """
    :param label_index: 需要预测的标签帧的索引
    :param label_path: 标签帧的路径
    :param save_path: 保存路径
    :param scene_name: 场景名称
    """
    mv = read_exr(os.path.join(label_path, f"{scene_name}WarpMotionVector.{str(label_index).zfill(4)}.exr"), channel=4)
    
    inpaint_mv = inpaint(mv)
    
    write_exr(os.path.join(save_path, f"{scene_name}WarpMotionVector.{str(label_index).zfill(4)}.exr"), inpaint_mv)

    return inpaint_mv


def main(root, sub_paths, debug=True):
    inpaint_init()
    save_path_ = "./fgsr_inpaint"

    # 获取数据元信息
    input_paths = []
    label_paths = []
    scene_names = []
    index_ranges = []
    for sub_path in sub_paths:
        input_path = os.path.join(root, sub_path)
        input_paths.append(input_path)
        scene_names.append(sub_path.split('/')[0])
        # input_path下所有文件名都是xxx.index.exr的格式，找到index的范围
        buffer_names = sorted(os.listdir(input_path))
        index_start = int(buffer_names[0].split('.')[1])
        index_end = int(buffer_names[-1].split('.')[1])
        
        index_ranges.append([index_start, index_end])

    for scene_name, input_path, index_range in zip(scene_names, input_paths, index_ranges):
        save_path = os.path.join(save_path_, scene_name)
        save_path = os.path.join(save_path, input_path.split('/')[-1])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for i in range(index_range[0], index_range[1] + 1, 2):
            inpaint_mv = inpaint_main(i, input_path, save_path, scene_name)
            
            print(save_path + ": " + str(i))

            # 如果处于debug模式，只循环三次
            if debug:
                if i >= index_range[0] + 5:
                    break


if __name__ == "__main__":
    root = "./fuse_me"
    sub_paths = ["Bunker/train2-60fps", "Bunker/test1-60fps"]
    # sub_paths = ["Bunker/train1-60fps"]
    main(root, sub_paths, debug=False)
