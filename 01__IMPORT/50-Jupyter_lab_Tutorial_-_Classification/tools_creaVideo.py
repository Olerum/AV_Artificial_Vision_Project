import numpy as np   # el módulo numpy se asocia con alias np
import time                                 # módulo de gestión del tiempo
import cv2           # el módulo de OpenCV
from IPython.display import clear_output, Image, display, HTML  # funcionalidad para mostrar imágenes en el cuaderno

##############################################################################################
# Definición de la clase Objeto2D
##############################################################################################
class Objeto2D(object):
    
    ##########################################################################################
    # Método constructor que se ejecuta durante la instanciación de un objeto
    def __init__(self, tipo='circles', pos=(0,0), pos_rand=(0,0), rot=(0,0), size=70, scala_rand=(1.0,1.0), color=(175,0,0), sigma=10, clase=0):
        self.tipo       = tipo
        self.size       = size       # 'circulo' es el radio, en 'cuadrado' es el lado
        self.scala_rand = scala_rand # rango de variaciones de escala para los dos ejes 
        self.pos        = pos        # pos
        self.pos_rand   = pos_rand   # desplazamiento aleatorio pos +- pos_rand
        self.rot        = rot        # (rot_min, rot_max) en grados
        self.color      = color  
        self.sigma      = sigma      # std del color
        self.clase      = clase      # indice sobre la tupla de objetos + 1
        self.visible    = True       # por defecto los objetos son visibles
        
        # parámetros dinámicos
        self.inc_pos   = (0,0)
        self.inc_color = (0,0,0)
        self.inc_rot   = (0,0)
        
    ##########################################################################################
    def next_pos(self):
        self.pos = (self.pos[0] + self.inc_pos[0], self.pos[1] + self.inc_pos[1])
        self.rot = (self.rot[0] + self.inc_rot[0], self.rot[1] + self.inc_rot[1])
    
    def next_color(self):
        self.color = (self.color[0] + self.inc_color[0], self.color[1] + self.inc_color[1], self.color[2] + self.inc_color[2])

##############################################################################################
# Definición de la clase Escena: conjunto de objetos
##############################################################################################
class Escena(object):
    # Método constructor que se ejecuta durante la instanciación de un objeto
    def __init__(self, objetos=(Objeto2D(pos=(144,180)),), num_objetos_visibles = 1):
        self.objetos = objetos
        self.num_objetos_visibles = num_objetos_visibles
        self.indices_objetos_visibles = []
        
    def clear(self):
        self.objetos = ()
        
    def add(self, objeto):
        self.objetos = self.objetos + (objeto, )
        if objeto.visible == True:
            self.indices_objetos_visibles.append(len(self.objetos)-1)
        
    def set_num_objetos_visibles(self, num):
        self.num_objetos_visibles = num

    def get_indices_objetos_visibles(self):
        return self.indices_objetos_visibles
                
    def set_objetos_visibles(self, indices):
        self.indices_objetos_visibles = []
        for k, obj in enumerate(self.objetos):
            if k in indices:
                obj.visible=True
                self.indices_objetos_visibles.append(k)
            else:
                obj.visible=False

    def set_objetos_invisibles(self, indices):
        self.indices_objetos_visibles = []
        for k, obj in enumerate(self.objetos):
            if k in indices:
                obj.visible=False
            else:
                obj.visible=True
                self.indices_objetos_visibles.append(k)
                
    def elige_objetos_visibles(self, indices):
        if indices[0] == -1:
            num_objetos_visibles = self.num_objetos_visibles
            num_objetos = len(self.objetos)
            
            # Si escena.num_objetos_visibles son todos, se muestran todos
            if num_objetos == num_objetos_visibles:
                indices = tuple(np.arange(num_objetos_visibles))
            else:
                indices = np.random.choice(self.indices_objetos_visibles, size = num_objetos_visibles)
                
        # Activa los objetos visibles
        self.set_objetos_visibles(indices)
        return indices
    
    def print_objetos_visibles(self):
        for k, obj in enumerate(self.objetos):
            if obj.visible:
                print(k,'visible')
            else:
                print(k,'invisible')
            
##############################################################################################
# Definición de la clase Camera
##############################################################################################
class Camera(object):
    
    ##########################################################################################
    # Método constructor que se ejecuta durante la instanciación de un objeto
    def __init__(self, img_fondo = np.zeros((288,360,1), dtype='uint8'), desenfoque = False, apply_homografia = False):
        self.img_fondo  = img_fondo
        self.img_size   = img_fondo.shape[:2]
        self.channels   = img_fondo.shape[2]
        self.num_pixels = self.img_size[0]*self.img_size[1]
        self.desenfoque = desenfoque
        self.homografia = self.calibraHomagrafia()
        self.apply_homografia = apply_homografia
    
    ##########################################################################################
    def proyect_object(self, obj):
        mask = np.zeros(self.img_size, dtype='uint8')
        # Se dibuja un circulo
        # https://docs.opencv.org/4.4.0/dc/da5/tutorial_py_drawing_functions.html
        if obj.tipo == 'circles':
            mask = cv2.circle(mask, obj.pos, obj.size, 1, -1)
            
        if obj.tipo == 'cuadrado':
            s0_half = int(obj.size[0]/2)
            s1_half = int(obj.size[1]/2)
            punto1 = (obj.pos[0]-s0_half, obj.pos[1]-s1_half)
            punto2 = (obj.pos[0]+s0_half, obj.pos[1]+s1_half)
            mask = cv2.rectangle(mask, punto1, punto2, 1, -1)
        
        if obj.tipo == 'digito':
            # genera un número aleatorio
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(mask, str(obj.clase), obj.pos, font, 9, 1, 9, cv2.LINE_AA)
            
        return mask

    ##########################################################################################
    def get_noise_channel(self,gris, sigma):
        img_random = np.random.normal(gris, sigma, self.num_pixels)
        img_random[img_random>255]=255
        img_random[img_random<0]=0
        img_random = np.uint8(np.reshape(img_random, self.img_size))
        return img_random

    ##########################################################################################
    def set_type_img_fondo(self, img_fondo):
        self.img_fondo  = img_fondo
        self.img_size   = img_fondo.shape[:2]
        self.channels   = img_fondo.shape[2]
        self.num_pixels = self.img_size[0]*self.img_size[1]
        
    def set_color_img_fondo(self, color=(50,50,50), sigma=0):
        for channel in range(self.channels):
            #print(channel)
            self.img_fondo[:,:,channel] = self.get_noise_channel(color[channel], sigma)

    ##########################################################################################
    def new_image(self, escena):
        
        # Copiamos la imagen de fondo original
        img = self.img_fondo.copy()
        
        for obj in escena.objetos:
            
            if obj.visible:
                # Actualiza objeto
                obj.next_pos()
                obj.next_color()

                mask = self.proyect_object(obj)

                # Escala la máscara del objeto
                if (obj.scala_rand[0] < obj.scala_rand[1]):
                    scala1 = np.random.uniform(obj.scala_rand[0], obj.scala_rand[1])
                    scala2 = np.random.uniform(obj.scala_rand[0], obj.scala_rand[1])
                    mask = self.scale(mask, (scala1, scala2))
                    
                # Rota la máscara del objeto
                if obj.rot[1]>obj.rot[0]:
                    rot = np.random.randint(obj.rot[0],high=obj.rot[1])
                else:
                    rot = obj.rot[0]
                mask = self.rotate(mask, rot)
                
                # Desplaza la mácara del objeto
                if obj.pos_rand[0] > 0:
                    d1 = np.random.randint(-obj.pos_rand[0],high=obj.pos_rand[0])
                else:
                    d1 = 0
                if obj.pos_rand[1] > 0:
                    d2 = np.random.randint(-obj.pos_rand[1],high=obj.pos_rand[1])
                else:
                    d2 = 0
                mask = self.translate(mask, (d1, d2))
                
                # Aplica una transformación de perspectiva
                if self.apply_homografia and (np.random.uniform() < 0.5):
                    mask = self.homografia(mask)

                for channel in range(self.channels):

                    # Crear la imagen del objeto y fondo
                    img_blob = np.multiply(mask, self.get_noise_channel(obj.color[channel], obj.sigma))
                    img_fondo = np.multiply(1-mask, img[:,:,channel])

                    # Fusionar con la imagen anterior
                    img[:,:,channel] = img_fondo + img_blob
                    
        # Aplicar un suavizado
        if self.desenfoque:
            img = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
        
        return img
    
    ##########################################################################################
    # Función de escalado de una imagen
    def scale(self, image, scala=(1.0, 1.0)):
        (h, w) = image.shape[:2]

        # realiza el escalado
        scale_matrix = np.float32([ [scala[0],0,0], [0,scala[1],0] ])
        scaled = cv2.warpAffine(image, scale_matrix, (w, h))

        return scaled
    
    # Función de traslación de una imagen
    def translate(self, image, desp=(0, 0)):
        (h, w) = image.shape[:2]

        # realiza la traslacion
        translation_matrix = np.float32([ [1, 0, desp[0] ], [0, 1, desp[1] ] ])
        translated = cv2.warpAffine(image, translation_matrix, (w, h))

        return translated

    # Función de rotación de una imagen
    def rotate(self, image, angle, center = None, scale = 1.0):
        (h, w) = image.shape[:2]

        if center is None:
            center = (w / 2, h / 2)

        # Perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        rotated = cv2.warpAffine(image, M, (w, h))

        return rotated

    # Función de homografia
    def calibraHomagrafia(self):
        pts1 = np.float32([[0,0],[360,0],[0,288],[360,288]])
        pts2 = np.float32([[np.random.randint(56),     np.random.randint(65)],
                           [360-np.random.randint(20), np.random.randint(52)],
                           [np.random.randint(28),     288-np.random.randint(50)],
                           [300-np.random.randint(50), 288-np.random.randint(100)]])
        return cv2.getPerspectiveTransform(pts1,pts2)
            
    def homografia(self, image):
        dst = cv2.warpPerspective(image, self.homografia, (image.shape[1], image.shape[0]))
        return dst


##############################################################################################
# Definición de la clase VideoVirtual
##############################################################################################                       
class VideoVirtual(object):
    
    ##########################################################################################
    # Método constructor que se ejecuta durante la instanciación de un objeto
    def __init__(self, tipo = 'circles', camera = Camera(), escena = Escena(), max_iter=14):
        self.tipo     = tipo
        self.iter     = 0
        self.max_iter = max_iter
        self.camera  = camera
        self.escena  = escena

        # Inicializa los tipos de escenarios
        if tipo == 'circles':
            # Crear la imagen de fondo de color
            img_fondo = np.zeros( (self.camera.img_size[0], self.camera.img_size[1], 1), dtype='uint8')
            self.camera.set_type_img_fondo(img_fondo)
            self.camera.set_color_img_fondo(color=(75,), sigma=10)
            self.camera.desenfoque = True
            
            # Inicializar los objetos de la escena
            self.escena.objetos = ()
            scale = 4.0 
            radio = int(np.sqrt( self.camera.num_pixels/(scale*np.pi) ) ) 
            centro = (int(self.camera.img_size[1]/2), int(self.camera.img_size[0]/2) )
            circulo = Objeto2D(tipo='circles', pos=centro, pos_rand=(10,10), size=radio, color=(200,0,0), sigma=10)
            circulo.inc_color = (-10,0,0)
            self.escena.add(circulo)
            
            # Activa todos los objetos como visibles
            self.escena.set_num_objetos_visibles(1)
            
        if tipo == 'colors':
            # Crear la imagen de fondo de color
            img_fondo = np.ones( (self.camera.img_size[0], self.camera.img_size[1], 3), dtype='uint8')*255
            self.camera.set_type_img_fondo(img_fondo)
            
            # definimos 8 restángulos y 8 colores
            colors = ((215,40,40),(40,215,40),(40,40,215),(215,215,40),(215,40,215),(40,215,215),(40,40,40),(215,215,215))
            num_rectangulos = len(colors)
            num_rectangulos_half = int(num_rectangulos/2)

            # Inicializar los objetos de la escena
            self.escena.objetos = ()
            inc_alto = int(self.camera.img_size[0]/2)
            inc_ancho = int(self.camera.img_size[1]/num_rectangulos_half)

            for k, bgr in enumerate(colors):

                i_alto = int(k / num_rectangulos_half) * inc_alto
                i_ancho = (k % num_rectangulos_half) * inc_ancho
                inc_ancho_half = int(inc_ancho/2)
                inc_alto_half  = int(inc_alto/2)
                centro = (i_ancho+inc_ancho_half, i_alto+inc_alto_half)
                
                cuadrado = Objeto2D(tipo='cuadrado', pos=centro, size=(inc_ancho, inc_alto), color=bgr, sigma=10)
                cuadrado.inc_color = (-3,+3,-3)
                self.escena.add(cuadrado)
                
            # Activa todos los objetos como visibles
            self.escena.set_num_objetos_visibles(8)

        if tipo == 'digits':
            # Crear la imagen de fondo de color
            img_fondo = np.zeros( (self.camera.img_size[0], self.camera.img_size[1], 3), dtype='uint8')
            self.camera.set_type_img_fondo(img_fondo)
            self.camera.set_color_img_fondo(color=[50, 50, 50], sigma=10)
            self.camera.desenfoque = True
            
            # Inicializar los objetos de la escena
            self.escena.clear()            
            centro = (int(self.camera.img_size[1]/4.0) , int(self.camera.img_size[0]/1.2) )
            for clase in range(10):
                digito = Objeto2D(tipo='digito', pos=centro, pos_rand=(10,10), rot=(-10,10), color=(0,200,0), sigma=10, clase=clase)
                self.escena.add(digito)
                
            # Activa todos los objetos como visibles
            self.escena.set_num_objetos_visibles(1)
                
        if tipo == 'figures':
            # Crear la imagen de fondo de color
            img_fondo = np.zeros( (self.camera.img_size[0], self.camera.img_size[1], 3), dtype='uint8')
            self.camera.set_type_img_fondo(img_fondo)
            self.camera.set_color_img_fondo(color=[50, 50, 50], sigma=10)
            
            # Inicializar los objetos de la escena
            self.escena.clear()
            
            # Clase 1: circulo
            scale = 4.0 
            radio = int(np.sqrt( self.camera.num_pixels/(scale*np.pi) ) ) 
            centro = (int(self.camera.img_size[1]/2), int(self.camera.img_size[0]/2) )
            circulo = Objeto2D(tipo='circles', pos=centro, pos_rand=(10,10), size=radio, scala_rand=(0.8,1.2), color=(0,0,255), sigma=10)
            self.escena.add(circulo)
            
            # Clase 2: cuadrado
            centro = (int(self.camera.img_size[1]/2) , int(self.camera.img_size[0]/2))
            inc_alto = int(self.camera.img_size[0]/2)
            inc_ancho = int(self.camera.img_size[1]/2)
            cuadrado = Objeto2D(tipo='cuadrado', pos=centro, pos_rand=(10,10), rot=(-5,5), size=(inc_ancho, inc_alto), scala_rand=(0.9,1.1), color=(0,0,255), sigma=10)
            self.escena.add(cuadrado)
                      
            # Activa todos los objetos como visibles
            self.escena.set_num_objetos_visibles(1)
                
    ##########################################################################################
    # Método para leer una imagen
    def read(self, indices=(-1,)):
        
        clase = 0
        img = []
            
        if self.iter < self.max_iter:
    
            self.iter = self.iter +1
        
            # Obtiene los objetos visibles iniciales
            #self.escena.print_objetos_visibles()
            indices_objetos_visibles = self.escena.get_indices_objetos_visibles()
            
            # Seleccionar los objetos entre los objetos_visibles
            indices = self.escena.elige_objetos_visibles(indices)
            clase = indices[0] + 1
            
            # Se genera una nueva imagen
            img = self.camera.new_image(self.escena)              

            # Hace visibles los objetos iniciales
            self.escena.set_objetos_visibles(indices_objetos_visibles)
            
        return clase, img
    
                    
    ##########################################################################################
    # Método para leer una tripleta de imágenes
    def read_triplet(self, indices=(-1,)):
        
        clase = 0
        img = []
            
        if self.iter < self.max_iter:
    
            self.iter = self.iter +1
            
            # Obtiene los objetos visibles iniciales
            #self.escena.print_objetos_visibles()
            indices_objetos_visibles = self.escena.get_indices_objetos_visibles()
            
            # Seleccionar objetos entre los objetos_visibles
            indices_positive = self.escena.elige_objetos_visibles(indices)
            clase = indices_positive[0] + 1
            
            # Se genera una nueva imagen
            img = self.camera.new_image(self.escena)
            
            # Se genera una imagen con los mismos objetos seleccionadas
            img_positive = self.camera.new_image(self.escena)
            
            # Se genera una imagen con objetos distintos a los seleccionados
            self.escena.set_objetos_invisibles(indices_positive)
            indices_negative = self.escena.elige_objetos_visibles(indices)
            img_negative = self.camera.new_image(self.escena)
            
            # Hace visibles los objetos iniciales
            self.escena.set_objetos_visibles(indices_objetos_visibles)

        return clase, img, img_positive, img_negative

    ##########################################################################################
    # Método para liberar la camara virtual de video
    def release(self):
        print('Number of generated images', self.iter)
