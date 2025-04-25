import numpy as np

def sample(texture, u, v) : 

    u = int(u * texture.shape[0])
    v = int((1-v) * texture.shape[1])
    return texture[v,u] / 255.0
    pass


class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data ):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.output = []

def edgeSide(p, v0, v1) : 
    return (p[0]-v0[0])*(v1[1]-v0[1]) - (p[1]-v0[1])*(v1[0]-v0[0])

class GraphicPipeline:
    def __init__ (self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))


    def VertexShader(self, vertex, data) :
        outputVertex = np.zeros((17))

        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0

        vec = np.array([[x],[y],[z],[w]])

        vec = np.matmul(data['projMatrix'],np.matmul(data['viewMatrix'],vec))

        outputVertex[0] = vec[0]/vec[3]
        outputVertex[1] = vec[1]/vec[3]
        outputVertex[2] = vec[2]/vec[3]

        outputVertex[3] = vertex[3]
        outputVertex[4] = vertex[4]
        outputVertex[5] = vertex[5]

        outputVertex[6] = data['cameraPosition'][0] - vertex[0]
        outputVertex[7] = data['cameraPosition'][1] - vertex[1]
        outputVertex[8] = data['cameraPosition'][2] - vertex[2]

        outputVertex[9] = data['lightPosition'][0] - vertex[0]
        outputVertex[10] = data['lightPosition'][1] - vertex[1]
        outputVertex[11] = data['lightPosition'][2] - vertex[2]

        outputVertex[12] = vertex[6]
        outputVertex[13] = vertex[7]

        outputVertex[14] = vertex[0]
        outputVertex[15] = vertex[1]
        outputVertex[16] = vertex[2]


        return outputVertex


    def Rasterizer(self, v0, v1, v2) :
        fragments = []

        #culling back face
        area = edgeSide(v0,v1,v2)
        if area < 0 :
            return fragments
        
        
        #AABBox computation
        #compute vertex coordinates in screen space
        v0_image = np.array([0,0])
        v0_image[0] = (v0[0]+1.0)/2.0 * self.width 
        v0_image[1] = ((v0[1]+1.0)/2.0) * self.height 

        v1_image = np.array([0,0])
        v1_image[0] = (v1[0]+1.0)/2.0 * self.width 
        v1_image[1] = ((v1[1]+1.0)/2.0) * self.height 

        v2_image = np.array([0,0])
        v2_image[0] = (v2[0]+1.0)/2.0 * self.width 
        v2_image[1] = (v2[1]+1.0)/2.0 * self.height 

        #compute the two point forming the AABBox
        A = np.min(np.array([v0_image,v1_image,v2_image]), axis = 0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis = 0)

        #cliping the bounding box with the borders of the image
        max_image = np.array([self.width-1,self.height-1])
        min_image = np.array([0.0,0.0])

        A  = np.max(np.array([A,min_image]),axis = 0)
        B  = np.min(np.array([B,max_image]),axis = 0)
        
        #cast bounding box to int
        A = A.astype(int)
        B = B.astype(int)
        #Compensate rounding of int cast
        B = B + 1

        #for each pixel in the bounding box
        for j in range(A[1], B[1]) : 
           for i in range(A[0], B[0]) :
                x = (i+0.5)/self.width * 2.0 - 1 
                y = (j+0.5)/self.height * 2.0 - 1

                p = np.array([x,y])
                
                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                #test if p is inside the triangle
                if (area0 >= 0 and area1 >= 0 and area2 >= 0) : 
                    
                    #Computing 2d barricentric coordinates
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area
                    
                    #one_over_z = lambda0 * 1/v0[2] + lambda1 * 1/v1[2] + lambda2 * 1/v2[2]
                    #z = 1/one_over_z
                    
                    z_ndc = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]
                    z = z_ndc * 0.5 + 0.5  


                    p = np.array([x,y,z])
                    
                    
                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = lambda0 * v0[3:] + lambda1 * v1[3:] + lambda2 * v2[3:]
                    
                    #Emiting Fragment
                    fragments.append(Fragment(i,j,z, interpolated_data))

        return fragments
    

    def fragmentShader(self, fragment, data):
        N = fragment.interpolated_data[0:3]
        N = N / np.linalg.norm(N)
        V = fragment.interpolated_data[3:6]
        V = V / np.linalg.norm(V)
        L = fragment.interpolated_data[6:9]
        L = L / np.linalg.norm(L)

        R = 2 * np.dot(L, N) * N - L
        ambient = 1.0
        diffuse = max(np.dot(N, L), 0)
        specular = np.power(max(np.dot(R, V), 0.0), 64)

        ka = 0.1
        kd = 1.0
        ks = 0.5
        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong * 4 + 1) / 6.0

        texture = sample(data['texture'], fragment.interpolated_data[9], fragment.interpolated_data[10])
        color = np.array([phong, phong, phong])

        # --- SHADOW ---
        fragPos = fragment.interpolated_data[11:14]
        vec = np.append(fragPos, 1.0).reshape((4, 1))

        light_space_pos = np.matmul(data['lightProjMatrix'], np.matmul(data['lightViewMatrix'], vec))
        light_space_pos /= light_space_pos[3]

        u = np.clip(light_space_pos[0, 0] * 0.5 + 0.5, 0, 1)
        v = np.clip(light_space_pos[1, 0] * 0.5 + 0.5, 0, 1)
        shadow_depth = light_space_pos[2, 0] * 0.5 + 0.5

        depthMap = data['shadowMap']
        height, width = depthMap.shape
        x = int(u * width)
        y = int((1 - v) * height)

        bias = 0.0005

        # --- PCF filtering ---
        shadow = 0
        samples = 3
        count = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                sx = int(x + dx)
                sy = int(y + dy)
                if 0 <= sx < width and 0 <= sy < height:
                    closest = depthMap[sy, sx]
                    if shadow_depth - bias > closest:
                        shadow += 1
                    count += 1
        shadow /= count

        color *= (1.0 - 0.5 * shadow)
        fragment.output = color


    def draw(self, vertices, triangles, data, shade=True):
        #Calling vertex shader
        self.newVertices = np.zeros((vertices.shape[0], 17))

        for i in range(vertices.shape[0]) :
            self.newVertices[i] = self.VertexShader(vertices[i],data)
        
        fragments = []
        #Calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))
        
        for f in fragments:
            if shade:
                self.fragmentShader(f, data)

            #depth test
            if self.depthBuffer[f.y][f.x] > f.depth : 
                self.depthBuffer[f.y][f.x] = f.depth
                if shade:
                    self.image[f.y][f.x] = f.output
                
            

