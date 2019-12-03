import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import cv2
import numpy as np
import pygame.locals
import pygame.display
import pygame.image
from glob import glob
import os
from utils import rotate
import time

class ObjLoader(object):
    def __init__(self, fileName):
        self.vertices = list()
        self.faces = list()
        try:
            file = open(fileName)
            for line in file:
                if line.startswith('v '):
                    line = line.strip().split()
                    vertex = (float(line[1]), float(line[2]), float(line[3]) )
                    vertex = (round(vertex[0], 2), round(vertex[1], 2), round(vertex[2], 2))
                    self.vertices.append(vertex)

                elif line.startswith('f'):
                    line = line.strip().split()
                    line[1].split('/')[0]
                    if '/' in line[1]:
                        line[1] = line[1].split('/')[0]
                    if '/' in line[2]:
                        line[2] = line[2].split('/')[0]
                    if '/' in line[3]:
                        line[3] = line[3].split('/')[0]
                    face = (int(line[1]), int(line[2]), int(line[3]) )
                    self.faces.append(face)
            file.close()
        except IOError:
            print(".obj file not found.")

    def render_scene(self):
        if len(self.faces) > 0:
            glRotatef(0.0, 0.0, 0.0, 1.0)
            glBegin(GL_TRIANGLES)
            #glBegin(GL_TRIANGLE_FAN)
            for face in self.faces:
                i = 0
                vertex1 = 0
                vertex2 = 0
                vertex3 = 0
                for f in face:
                    vertexDraw = self.vertices[int(f) - 1]
                    #make vertex set in face
                    if(i == 0):
                        vertex1 = vertexDraw
                        i += 1
                    elif(i == 1):
                        vertex2 = vertexDraw
                        i += 1
                    elif(i == 2):
                        vertex3 = vertexDraw
                        i = 0
                #make vectors
                vector1 = np.subtract(vertex2, vertex1)
                vector2 = np.subtract(vertex3, vertex1)
                #Compute normal
                normal = np.cross(vector1, vector2)
                #Compute normalized sum
                divider = math.sqrt(normal[0] * normal[0] + normal[1]*normal[1] + normal[2]*normal[2])
                glColor4f(math.fabs(normal[0]/divider), math.fabs(normal[1]/divider), math.fabs(normal[2]/divider), 1.0)
                glVertex3fv(vertex1)
                glVertex3fv(vertex2)
                glVertex3fv(vertex3)
            glEnd()

class objItem(object):
    def __init__(self, obj_path):
        self.angle = 0
        self.vertices = []
        self.faces = []
        # self.coordinates = [-130, -450, -1300]  # [x,y,z]
        self.coordinates = [0,0,0]
        self.obj = ObjLoader(obj_path)
        self.position = [0, 0, -50]

    def render_scene(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glShadeModel(GL_SMOOTH);
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(0, 0, 0, math.sin(math.radians(self.angle)), 0, math.cos(math.radians(self.angle)) * -1, 0, 1, 0)
        glTranslatef(self.coordinates[0], self.coordinates[1], self.coordinates[2])

    def move_forward(self):
        self.coordinates[2] += 10 * math.cos(math.radians(self.angle))
        self.coordinates[0] -= 10 * math.sin(math.radians(self.angle))

    def move_back(self):
        self.coordinates[2] -= 10 * math.cos(math.radians(self.angle))
        self.coordinates[0] += 10 * math.sin(math.radians(self.angle))

    def move_left(self, n):
        self.coordinates[0] += n * math.cos(math.radians(self.angle))
        self.coordinates[2] += n * math.sin(math.radians(self.angle))

    def move_right(self, n):
        self.coordinates[0] -= n * math.cos(math.radians(self.angle))
        self.coordinates[2] -= n * math.sin(math.radians(self.angle))
        print(self.coordinates[0])
        print(self.coordinates[2])
    def rotate(self, n):
        self.angle += n

    def fullRotate(self):
        for i in range(0, 36):
            self.angle += 10
            self.move_left(10)
            self.render_scene()
            self.obj.render_scene()
            pygame.display.flip()


## Data Preparation
class ObjPreparation(object):
    def __init__(self, base_folder, size, angle_list):
        self.base_folder = base_folder

        self.input_list = glob(os.path.join(self.base_folder, '*.obj'))
        self.input_list = [inp.replace('\\', '/') for inp in self.input_list]
        self.size = size

        self.angle_list = angle_list

    def pre_processing(self):
        for input in self.input_list:
            self.output = os.path.join(self.base_folder, 'processed', input.split('/')[-1]).replace('\\', '/')
            self.process_obj(input, self.output, size=self.size)

            for ix, angle in enumerate(self.angle_list):
                self.rotated_folder = '/'.join(self.output.replace('processed', 'rotated').split('/')[:-1])
                self.rotated_path = os.path.join(self.rotated_folder, self.output.split('/')[-1].replace('.obj', '') + '_%03d.obj' %ix)
                self.rotate(self.output, self.rotated_path, angle)

    def rotate(self, obj_in, obj_out, angle):
        rotate(obj_in, obj_out, angle, base=False, random=False)

    def process_obj(self, obj_in, obj_out, size=5):
        f_open = open(obj_in)
        f_save = open(obj_out, 'w')
        data = f_open.readlines()
        vx_list, vy_list, vz_list = [], [], []
        for d in data:
            d_split = d.split(' ')
            if d_split[0] == 'v':
                vx_list.append(float(d_split[1]))
                vy_list.append(float(d_split[2]))
                vz_list.append(float(d_split[3][:-1]))

        vx_mid = (max(vx_list) + min(vx_list)) / 2
        vy_mid = (max(vy_list) + min(vy_list)) / 2
        vz_mid = (max(vz_list) + min(vz_list)) / 2

        len_vx = abs(max(vx_list) - min(vx_list))
        len_vy = abs(max(vy_list) - min(vy_list))
        len_vz = abs(max(vz_list) - min(vz_list))

        len_diag = math.sqrt(len_vx**2 + len_vy**2 + len_vz**2)

        for d in data:
            d_split = d.split(' ')
            if d_split[0] == 'v':
                vx = float(d_split[1])
                vy = float(d_split[2])
                vz = float(d_split[3][:-1])

                vx = vx - vx_mid
                vy = vy - vy_mid
                vz = vz - vz_mid

                vx = size * (vx / len_diag)
                vy = size * (vy / len_diag)
                vz = size * (vz / len_diag)

                d = 'v ' + str(vx) + ' ' + str(vy) + ' ' + str(vz) + '\n'
            f_save.write(d)

        f_open.close()
        f_save.close()

def rotate_all():
    base_x = [[0,0,0], [0,90,0], [0,180,0], [0,270,0]] # y, x, z
    base_z = [[0,0,0], [0,0,90], [0,0,180]]
    base_y = [[0,0,0], [90,0,0]]

    base_lists = [np.asarray(x)+np.asarray(z)+np.asarray(z) for x in base_x for z in base_z for y in base_y]
    rotate_lists = [np.asarray(x_) + np.asarray(z_) for x_ in [[0,10,0], [0,40,0], [0,70,0]] for z_ in [[0,0,10], [0,0,40], [0,0,70]]]

    angle_list = [(base_ + rotate_).tolist() for base_ in base_lists for rotate_ in rotate_lists]
    return angle_list

def main(furniture_folder, show=False):
    ## Preparation for data (obj)
    for folder in ['processed', 'outline', 'rotated', 'surface']:
        os.makedirs(os.path.join(furniture_folder, folder), exist_ok=True)

    angle_list = rotate_all()
    obj_tool = ObjPreparation(furniture_folder, angle_list= angle_list, size=5)
    obj_tool.pre_processing()

    width = 512
    height = 512

    rotated_list = glob(os.path.join(furniture_folder, 'rotated', '*.obj'))
    print('Loading the rotated obj for surface normal rendering')

    for rotated in rotated_list:
        ## Initialize
        pygame.init()
        pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.locals.GL_MULTISAMPLESAMPLES, 16)
        window = pygame.display.set_mode((width, height), pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("Python furniture module")
        clock = pygame.time.Clock()

        # Function checker
        glDisable(GL_TEXTURE_2D)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45.0, float(width) / height, .1, 10000.)
        glMatrixMode(GL_MODELVIEW)

        ## Surface Normal Generation
        surface_name = rotated.replace('rotated', 'surface').replace('.obj', '.png')
        meshObj = objItem(rotated)
        meshObj.move_back()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        meshObj.render_scene()
        meshObj.obj.render_scene()
        pygame.display.flip()
        pygame.image.save(window, surface_name)
        pygame.quit()

        ## Outline Generation
        outline_name = surface_name.replace('surface','outline')
        src = cv2.imread(surface_name, cv2.IMREAD_GRAYSCALE)

        laplacian = cv2.Laplacian(src, cv2.CV_8U, ksize=5)

        dst = cv2.bitwise_not(laplacian)
        dst = cv2.GaussianBlur(dst, (5, 5), 0)

        cv2.imwrite(outline_name, dst)

        if show == True:
            cv2.imshow('dst', dst)
            cv2.waitKey(1)
            time.sleep(0.5)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    base_folder = 'C:/data/external_imp'
    furniture_list = os.listdir(base_folder)
    show = False
    for furniture in furniture_list:
        furniture_folder = os.path.join(base_folder, furniture)
        main(furniture_folder, show=show)