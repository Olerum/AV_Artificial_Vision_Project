import numpy as np
import cv2
import os
from PIL import Image
from torchvision.transforms import transforms
from math import pi
import math
import random
import matplotlib.pyplot as plt

"""Data normalization"""
def normal(M):

    nPuntos = len(M[0])
    if len(M) == 3:
        puntos = np.concatenate(([M[0] / M[2]], [M[1] / M[2]]))
        centroid = np.array([[sum(puntos[0])], [sum(puntos[1])]]) / nPuntos
        resta = np.concatenate(([puntos[0] - centroid[0]], [puntos[1] - centroid[1]]))
        dista = np.diag(np.dot(np.transpose(resta), resta))
        dista2 = np.sqrt(dista)        
        escala = sum(dista2) / nPuntos
        T = np.concatenate((np.concatenate((np.identity(len(puntos)) / escala, np.array([[0, 0]]))),
                            np.concatenate((centroid / -escala, np.array([[1]])))), axis=1)
        N = np.dot(T, M)
    if len(M) == 4:
        puntos = np.concatenate(([M[0] / M[3]], [M[1] / M[3]], [M[2] / M[3]]))
        centroid = np.array([[sum(puntos[0])], [sum(puntos[1])], [sum(puntos[2])]]) / nPuntos
        resta = np.concatenate(([puntos[0] - centroid[0]], [puntos[1] - centroid[1]], [puntos[2] - centroid[2]]))
        dista = np.diag(np.dot(np.transpose(resta), resta))
        dista2 = np.sqrt(dista)        
        escala = sum(dista2) / nPuntos
        T = np.concatenate((np.concatenate((np.identity(len(puntos)) / escala, np.array([[0, 0, 0]]))),
                            np.concatenate((centroid / -escala, np.array([[1]])))), axis=1)
        N = np.dot(T, M)
    return N, T

"""Estimation of the projection matrix"""
def minEst(X, U):

    a = np.zeros((2 * len(X[0]), 11))
    b = np.zeros((2 * len(X[0]), 1))
    for i in range(2 * len(X[0])):
        for j in range(12):
            if i % 2 == 0:
                if j == 0:
                    a[i, j] = X[0][int(i / 2)]
                elif j == 1:
                    a[i, j] = X[1][int(i / 2)]
                elif j == 2:
                    a[i, j] = X[2][int(i / 2)]
                elif j == 3:
                    a[i, j] = X[3][int(i / 2)]
                elif j == 8:
                    a[i, j] = -U[0][int(i / 2)] * X[0][int(i / 2)]
                elif j == 9:
                    a[i, j] = -U[0][int(i / 2)] * X[1][int(i / 2)]
                elif j == 10:
                    a[i, j] = -U[0][int(i / 2)] * X[2][int(i / 2)]
                elif j == 11:
                    b[i, 0] = U[0][int(i / 2)] * X[3][int(i / 2)]
            else:
                if j == 4:
                    a[i, j] = X[0][int((i - 1) / 2)]
                elif j == 5:
                    a[i, j] = X[1][int((i - 1) / 2)]
                elif j == 6:
                    a[i, j] = X[2][int((i - 1) / 2)]
                elif j == 7:
                    a[i, j] = X[3][int((i - 1) / 2)]
                elif j == 8:
                    a[i, j] = -U[1][int((i - 1) / 2)] * X[0][int(i / 2)]
                elif j == 9:
                    a[i, j] = -U[1][int((i - 1) / 2)] * X[1][int(i / 2)]
                elif j == 10:
                    a[i, j] = -U[1][int((i - 1) / 2)] * X[2][int(i / 2)]
                elif j == 11:
                    b[i, 0] = U[1][int((i - 1) / 2)] * X[3][int(i / 2)]
  
    C = np.linalg.inv(np.dot(np.transpose(a), a))
    P = np.dot(np.dot(C, np.transpose(a)), b)
    
    E = b - np.dot(a, P)
    nPuntos = len(X)
    varianza = np.dot(np.transpose(E), E) / (2 * nPuntos - 11)
    covarza = varianza[0][0] * C
    covarzaN = np.concatenate((np.concatenate((covarza, np.zeros((11, 1))), axis=1), np.zeros((1, 12))))
    PestN = np.concatenate((P, np.array([[1]])))
    return PestN, covarzaN


"""Extraction of parameters from the estimated projection matrix"""
def param(P, varianza):

    lambd = math.sqrt(np.dot(np.transpose(P[8:11]), P[8:11]))
    p3V = varianza[8:11, 8:11]
    lambdaV = np.dot(np.dot(np.transpose(P[8:11]), p3V), P[8:11]) / (4 * np.dot(np.transpose(P[8:11]), P[8:11]))
    Pn = P / lambd
    Pn1V = (1 / (lambd ** 4)) * (P * lambdaV * np.transpose(P))
    Pn2V = varianza / (lambd ** 2)
    PnV = Pn1V + Pn2V

    p1 = Pn[0:3]
    p14 = Pn[3]
    p2 = Pn[4:7]
    p24 = Pn[7]
    p3 = Pn[8:11]
    p34 = Pn[11]

    P1V = PnV[0:3, 0:3]
    P14V = PnV[3][3]
    P2V = PnV[4:7, 4:7]
    P24V = PnV[7][7]
    P3V = PnV[8:11, 8:11]
    P34V = PnV[11][11]

    tz = p34
    tzV = P34V

    r3 = p3
    r3V = P3V

    uo = np.dot(np.transpose(p1), p3)
    uoV = np.dot(np.dot(np.transpose(p3), P1V), p3) + np.dot(np.dot(np.transpose(p1), P3V), p1)

    vo = np.dot(np.transpose(p2), p3)
    voV = np.dot(np.dot(np.transpose(p3), P2V), p3) + np.dot(np.dot(np.transpose(p2), P3V), p2)

    fu = math.sqrt(np.dot(np.transpose(p1), p1) - uo ** 2)
    fuV = (np.dot(np.dot(np.transpose(p1), P1V), p1) + np.dot(np.dot(uo, uoV), uo)) / (
                np.dot(np.transpose(p1), p1) - uo ** 2)

    fv = math.sqrt(np.dot(np.transpose(p2), p2) - vo ** 2)
    fvV = (np.dot(np.dot(np.transpose(p2), P2V), p2) + np.dot(np.dot(vo, voV), vo)) / (
                np.dot(np.transpose(p2), p2) - vo ** 2)

    tx = (p14 - np.dot(uo, p34)) / fu
    txV = (P14V + (p34 ** 2) * uoV + (uo ** 2) * P34V + ((((p14 - uo * p34) ** 2) * fuV) / fu ** 2)) / fu ** 2

    ty = (p24 - np.dot(vo, p34)) / fv
    tyV = (P24V + (p34 ** 2) * voV + (vo ** 2) * P34V + ((((p24 - vo * p34) ** 2) * fvV) / fv ** 2)) / fv ** 2

    r1 = (p1 - uo * p3) / fu

    r2 = (p2 - vo * p3) / fv

    Pm = np.concatenate((np.array([[fu]]), np.array([[fv]]), uo, vo, np.array([tx]), np.array([ty]), np.array([tz])))
    Pvar = np.concatenate((fuV, fvV, uoV, voV, txV, tyV, np.array([[tzV]])))
    return Pm, Pvar

"""Functionality to visualize matrices"""
UPPER_LEFT = u'\u250c'
UPPER_RIGHT = u'\u2510'
LOWER_LEFT = u'\u2514'
LOWER_RIGHT = u'\u2518'
HORIZONTAL = u'\u2500'
VERTICAL = u'\u2502'

def upper_line(width):
    return UPPER_LEFT + HORIZONTAL * width + UPPER_RIGHT

def lower_line(width):
    return LOWER_LEFT + HORIZONTAL * width + LOWER_RIGHT

def left_line(height):
    return "\n".join([UPPER_LEFT] + [VERTICAL] * height + [LOWER_LEFT])

def right_line(height):
    return "\n".join([UPPER_RIGHT] + [VERTICAL] * height + [LOWER_RIGHT])

def ndtotext(A, w=None, h=None):
    """Returns a string to pretty print the numpy.ndarray `A`.

    Currently supports 1 - 3 dimensions only.
    Raises a NotImplementedError if an array with more dimensions is passed.

    Describe `w` and `h`.
    """
    if A.ndim == 1:
        if w is None:
            #return str(A)
            return np.array2string(A, formatter={'float_kind':lambda A: "%.2f" % A})
        #s = " ".join([str(value).rjust(width) for value, width in zip(A, w)])
        s = " ".join([np.array2string(value, formatter={'float_kind':lambda value: "%.2f" % value}).rjust(width) for value, width in zip(A, w)])
        return '[{}]'.format(s)
    elif A.ndim == 2:
        #widths = [max([len(str(s)) for s in A[:, i]]) for i in range(A.shape[1])]
        widths = [max([len(np.array2string(s, formatter={'float_kind':lambda s: "%.2f" % s})) for s in A[:, i]]) for i in range(A.shape[1])]
        s = "".join([' ' + ndtotext(AA, w=widths) + ' \n' for AA in A])
        w0 = sum(widths) + len(widths) - 1 + 2 # spaces between elements and corners
        return upper_line(w0) + '\n'  + s + lower_line(w0)
    elif A.ndim == 3:
        h = A.shape[1]
        strings = [left_line(h)]
        strings.extend(ndtotext(a) + '\n' for a in A)
        strings.append(right_line(h))
        return '\n'.join(''.join(pair) for pair in zip(*map(str.splitlines, strings)))
    raise NotImplementedError("Currently only 1 - 3 dimensions are supported")

"""Composition of the matrix of extrinsic parameters"""
def compose_extrinsec_matrix(radio, angulo):
    
    rX = (pi / 180) * (angulo)  # angulo de rotación en el eje X
    rY = 0  # angulo de rotación en el eje Y
    rZ = 0  # angulo de rotación en el eje Z
    tX = 0  # Traslacion en el eje X
    tY = math.sin((pi / 180) * (angulo)) * radio  # Traslacion en el eje Y
    tZ = -math.cos((pi / 180) * (angulo)) * radio  # Traslacion en el eje Z
    
    Rx = np.array([[1, 0, 0], [0, math.cos(rX), -math.sin(rX)],
                   [0, math.sin(rX), math.cos(rX)]])  # matriz de traslación según rotación en el eje X
    Ry = np.array([[math.cos(rY), 0, math.sin(rY)], [0, 1, 0],
                   [-math.sin(rY), 0, math.cos(rY)]])  # matriz de traslación según rotación en el eje Y
    Rz = np.array([[math.cos(rZ), -math.sin(rZ), 0], [math.sin(rZ), math.cos(rZ), 0],
                   [0, 0, 1]])  # matriz de traslación según rotación en el eje Z
    R = np.dot(np.dot(Rx, Ry), Rz)  # Matriz de rotación que da la orientación de la camara en el espacio 3D
    T = np.array([[tX], [tY], [tZ]])  # Matriz de traslacion que situa la camara en el espacio 3D
    
    cTw = np.linalg.inv(np.concatenate((np.concatenate((R, T), axis=1), np.array([[0, 0, 0, 1]]))))
    return cTw

"""Composition of the projection matrix"""
def compose_P(fu, fv, uo, vo, radio, angulo):

    T = compose_extrinsec_matrix(radio, angulo)
    K = np.array([[fu, 0, uo], [0, fv, vo], [0, 0, 1]])
    K1 = np.array([[fu, 0, uo, 0], [0, fv, vo, 0], [0, 0, 1, 0]])

    P = np.dot(K1,T)
    P=P/P[2][3]
    #print('Projection matrix:\n',P)
    return P

"""Camera class"""
class Cam:
    
    """Intrinsic parameters """
    fu = 1580  # fu = f/dx
    fv = 1580  # fv = f/dy
    uo = 350  # uo = xo
    vo = 300  # vo = yo

    """Extrinsic parameters"""
    radio = 1000
    angulo = 180 - 45
    
    """Camera class constructor"""
    def __init__(self, *args): #_fu, _fv, _uo, _vo, _radio, _angulo

        if args:
            self.fu = args[0]
            self.fv = args[1]
            self.uo = args[2]
            self.vo = args[3]
            self.radio = args[4]
            self.angulo = args[5]
        
        self.P = compose_P( self.fu, self.fv, self.uo, self.vo, self.radio, self.angulo)
        print('Projection matrix:')
        print(ndtotext(self.P))
           
    """Returns the pixels U corresponding to the projections of the world points X with the camera model P"""
    def project(self, *args):
        if args:
            if len(args) == 1:
                P = self.P
                X =  args[0]
            elif len(args) == 2:
                P =  args[0]
                X =  args[1]
            else:
                return(-1)
            U = np.dot(P, X)
            U = np.concatenate([[U[0] / U[2]], [U[1] / U[2]], [np.ones(U[0].shape)]])
            return(U)
        return(-1)
    
    """Show image"""
    def im_show(self, U):
        nPuntos = len(U[0])
        nPuntos_mitad = int(nPuntos/2)
        fig = plt.figure(figsize=(8,6))
        plt.plot(U[0][:nPuntos_mitad], U[1][:nPuntos_mitad], marker='s', ls='');
        plt.plot(U[0][nPuntos_mitad:], U[1][nPuntos_mitad:], marker='^', ls='');

    """Defines the homography of the XY plane of the world"""
    def get_Homografy_planeXY(self):

        T = compose_extrinsec_matrix(self.radio, self.angulo)   
        T = np.delete(T,2, axis=1)
        T = np.delete(T,3, axis=0)
        K = np.array([[self.fu, 0, self.uo], [0, self.fv, self.vo], [0, 0, 1]])

        H = np.dot(K, T)
        H=H/H[2][2]
        #print('Homography matrix (XY plane):\n',H)
        return H
    
    """Gets the array of intrinsic parameters"""
    def get_camera_matrix(self):
        return np.array([[self.fu, 0, self.uo], [0, self.fv, self.vo], [0, 0, 1]])
    
    """Gets the array of extrinsic parameters"""
    def get_extrinsec_matrix(self):
        return compose_extrinsec_matrix(self.radio, self.angulo)