# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:42:07 2021

@author: user
"""


import json,os,pyproj,urllib.request,cv2
import numpy as np
import json
from PIL import Image
from io import BytesIO
from pyproj import Proj, transform
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point,LineString,Polygon
from matplotlib.pyplot import imread
from pyproj import Transformer
import copy,operator
import math

geod = pyproj.Geod(ellps='WGS84')

def dist_n(p0,p1):
    a,b= Point(p0),Point(p1)
    angle1,angle2,distance1 = geod.inv(a.x, a.y, b.x, b.y)
    return distance1

def plot_line(ax, ob,color , w = 0.5,alpha =1):
    x, y = ob.xy
    ax.plot(x, y, color=color, linewidth=w, alpha =1, solid_capstyle='round', zorder=1)

def formart_point(point):
    return (float("{:.5f}".format(point[0])), float("{:.5f}".format(point[1])))

        
def makeRequest(rq):
    f = urllib.request.urlopen(rq)
    data = f.read()
    return data

def find_folder(direction_out):
    try:
        if os.path.isdir(direction_out):
            os.remove(direction_out)
        os.mkdir(direction_out)
    except:
        pass   
    
      
def new_directions(direct,name):    
    find_folder(name) 
    directions={}        
    for d in direct.keys():
        directions[d]=name+'\\'+direct[d]   
    return directions

def count_step_and_view(dist,min_dist,radius):
#    if dist//min_dist>10:
#        step=min_dist*10
#        view=radius*5
#    else:
    step,view=min_dist,radius
    return step,view


def check_path(path):
    if len(path)<2:
        return False
    else:
        for point in path:            
            if type(point) is not list or len(point)!=2:
                return False
        return True
    
def set_points(start,finish,min_dist,radius):        
    dist = dist_n(start,finish)
    step,view = count_step_and_view(dist,min_dist,radius)
    delta= int(dist//step)+1
    if delta==1:
        delta=2
    return dist,step,view, delta

def get_list_points(start,finish,delta,sis_coord):
    transformer    = Transformer.from_crs(sis_coord['WGS_84'],sis_coord['Pseudo_Mercator'])     
    transformer_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'],sis_coord['WGS_84'])   
    l =[]
    point0 = transformer.transform(start[1],start[0])
    point1 = transformer.transform(finish[1],finish[0])
    xquery = np.linspace(point0[0],point1[0], delta)
    yquery = np.linspace(point0[1],point1[1], delta)
    for j in range(len(xquery)):
        p = transformer_84.transform(xquery[j],yquery[j])
        new_p=formart_point(p)
        l.append((new_p[1],new_p[0]))
    return l

def get_list_points_mult_direction (path,min_dist,radius,sis_coord):
    Point_WGS84=[formart_point((path[0][0],path[0][1]))]
    V=[]
    for i in range(len(path)-1):
        start,finish = formart_point(path[i]),formart_point(path[i+1])
        dist,step,view, delta = set_points(start,finish,min_dist,radius)
        V.append(view)
        l = get_list_points(start,finish,delta,sis_coord)
        Point_WGS84.extend(l[1:])
    return V,Point_WGS84

def extend_path(start,finish, step_length,remain,dist,sis_coord):
    transformer    = Transformer.from_crs(sis_coord['WGS_84'],sis_coord['Pseudo_Mercator'])     
    transformer_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'],sis_coord['WGS_84'])       
    dist_to_new_point = int(step_length-remain)+1            
    point0 = transformer.transform(start[1],start[0])
    point1 = transformer.transform(finish[1],finish[0])
    detX = (point1[0]-point0[0])/dist
    delY = (point1[1]-point0[1])/dist            
    new_finish = (point1[0]+(dist_to_new_point*detX),point1[1]+(dist_to_new_point*delY))
    n_finish = transformer_84.transform(new_finish[0],new_finish[1])
    return (n_finish[1],n_finish[0])


def list_of_points(path,direct,step_length,shooting_radius,sis_coord ):
    if len(path)==2: 
        start,finish = formart_point(path[0]),formart_point(path[1])
        name = str(path[0])+'_'+str(path[1])
        directions=new_directions(direct,name)        
        dist,step,view, delta = set_points(start,finish,step_length,shooting_radius)
        remain = dist%step_length
        if remain!=0: 
            new_finish=extend_path(start,finish, step_length,remain,dist,sis_coord)
            dist,step,view, delta = set_points(start,new_finish,step_length,shooting_radius)
            Point_WGS84 = get_list_points(start,new_finish,delta,sis_coord)
        else:
            Point_WGS84 = get_list_points(start,finish,delta,sis_coord)
    else:
        name = str([path[0][0],path[0][1]])+'_'+str([path[-1][0],path[-1][1]])
        directions=new_directions(direct,name)   
        step_length,shooting_radius=100,50
        V,Point_WGS84 = get_list_points_mult_direction (path,step_length,shooting_radius,sis_coord)
        view = V[0]
    return directions,view,Point_WGS84,name



def latlon_data (start,finish,binKey):    
    lat_max,lat_min = max([start[1],finish[1]]),min([start[1],finish[1]])
    lon_max,lon_min = max([start[0],finish[0]]),min([start[0],finish[0]])
    rq = "http://dev.virtualearth.net/REST/V1/Imagery/Map/AerialWithLabels?mapArea={0},{1},{2},{3}&ms=1000,100&mmd=0&key={4}"
    rq = "http://dev.virtualearth.net/REST/V1/Imagery/Map/AerialWithLabels?mapArea={0},{1},{2},{3}&ms=1000,1000&mmd={4}&key={5}"
    rq1 = rq.format(lat_min, lon_min, lat_max, lon_max, 0, binKey)
    rq2 = rq.format(lat_min, lon_min, lat_max, lon_max, 1, binKey)
    return rq1,rq2


def get_bing_tile(start,finish,binKey):
    rq1,rq2= latlon_data (start,finish, binKey)  
    tile = makeRequest(rq1)
    retJson = makeRequest(rq2).decode('utf-8')
    obj = json.loads(retJson)
    zoom = int(obj['resourceSets'][0]['resources'][0]['zoom'])
#    print('Zoom: ',zoom)
    coords = obj['resourceSets'][0]['resources'][0]['bbox']
    size_picture = (int(obj['resourceSets'][0]['resources'][0]['imageWidth']),int(obj['resourceSets'][0]['resources'][0]['imageHeight']))
    return [coords[1],coords[3],coords[0],coords[2]], size_picture,zoom,tile


def get_tile_dict(Point_WGS84,binKey):  
#    fig, ax = plt.subplots(num=None, figsize=(24,12),  dpi=30)
    Tiles,Z={},{}
    for i in range(len(Point_WGS84)-1):
        coords, size_picture,zoom,tile = get_bing_tile(Point_WGS84[i],Point_WGS84[i+1],binKey)
#        print(zoom)
        Tiles[(Point_WGS84[i],Point_WGS84[i+1])]={'box': [(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2])],
                                   'size': size_picture,
                                   'del_width': dist_n((coords[0],coords[2]),(coords[1],coords[2]))/size_picture[0],
                                   'del_high':  dist_n((coords[0],coords[2]),(coords[0],coords[3]))/size_picture[1], 
                                   'polygon':Polygon([(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2]),(coords[0],coords[2])]),
                                   'line':   LineString([Point_WGS84[i],Point_WGS84[i+1]]),
                                   'start':  Point_WGS84[i],
                                   'finish': Point_WGS84[i+1],
                                   'zoom': zoom,
                                   'tile': tile,
                                   'id': i}
#        ax.add_patch(PolygonPatch(Polygon([(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2]),(coords[0],coords[2])]), fc='grey', ec='grey', alpha=0.2, zorder=1))              
#        plot_line(ax, LineString([Point_WGS84[i],Point_WGS84[i+1]]) ,'red' , w = 5,alpha =1)
        Z[(Point_WGS84[i],Point_WGS84[i+1])]=zoom
#    plt.show()
#    plt.close()
    return Tiles,Z

def check_zoom(Tiles,Z,all_zooms,Point_WGS84,binKey,sis_coord):
    Del_tiles,T=[],{}
    popular_zoom = max(all_zooms.items(), key=operator.itemgetter(1))[0]
    for i in Tiles.keys():
        if Tiles[i]['zoom']!=popular_zoom:
            Del_tiles.append(i)            
            point0,point1 = Tiles[i]['start'],Tiles[i]['finish']
            New_points=get_list_points(point0,point1,3,sis_coord)
            Point_WGS84.insert (Point_WGS84.index(i[0])+1, New_points[1])
            for j in range(len(New_points)-1):
                coords, size_picture,zoom,tile = get_bing_tile(New_points[j],New_points[j+1],binKey) 
                T[(New_points[j],New_points[j+1])]={'box': [(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2])],
                                              'size': size_picture,
                                              'del_width': dist_n((coords[0],coords[2]),(coords[1],coords[2]))/size_picture[0],
                                              'del_high':  dist_n((coords[0],coords[2]),(coords[0],coords[3]))/size_picture[1], 
                                              'polygon':Polygon([(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2]),(coords[0],coords[2])]),
                                              'line':   LineString([New_points[j],New_points[j+1]]),
                                              'start':  New_points[j],
                                              'finish': New_points[j+1],
                                              'zoom': zoom,
                                              'tile': tile,
                                              'id': (Tiles[i]['id'],j)}                
                Z[(point0,point1)]=zoom
    if len(Del_tiles)!=0:
        for d in Del_tiles:
            del Tiles[d],Z[d]
    T1 = {**Tiles, **T}
    return T1,Z,Point_WGS84


def path_plan(Tiles, Point_WGS84, ax):
    L=[]
    for i in  range(len(Point_WGS84)-1):
        points=(Point_WGS84[i],Point_WGS84[i+1])
        if points in Tiles.keys():
            L.append(points)
            plot_line(ax, Tiles[points]['line'] ,'red' , w = 1,alpha =1)
            ax.plot(Tiles[points]['start'][0],Tiles[points]['start'][1],'o', markersize=7, color='red') 
            ax.plot(Tiles[points]['finish'][0],Tiles[points]['finish'][1],'o', markersize=6, color='blue')
            ax.add_patch(PolygonPatch(Tiles[points]['polygon'], fc='gray', ec='gray', alpha=0.2, zorder=1))
#            ax.text(Tiles[points]['start'][0],Tiles[points]['start'][1], str(i), ha='center', va='center',color='black',fontsize=40)     
        else:
            raise Exception ('error', points)
    ax.set_aspect(1)    
    return L,ax

def save_tile_as_picture(Tiles,direction_output,name):   
    id_pict=0
    for key in Tiles.keys():
        image = Image.open(BytesIO(Tiles[key]['tile']))
        find_folder(direction_output)
        path = os.path.join(direction_output,'tile_satellite('+name+'_'+str(id_pict)+').jpg')
        image.save(path)
        Tiles[key]['name_pict'] = name+'_'+str(id_pict)+')'
        id_pict+=1
    return Tiles




def get_angle(point0,point1):
    P1 = Point(point1)
    P2 = Point((point0[0],point1[1]))
    interP = Point(point0)
        
    dx = P1.x-interP.x
    dy = P1.y-interP.y
            
    dx2 = interP.x-P2.x
    dy2 = interP.y-P2.y
            
    azimuth1 = np.arctan2(dx,dy)*180/np.pi
    azimuth2 = np.arctan2(dx2,dy2)*180/np.pi
     
    azimuth11 = 360-azimuth1    # dx>0 & dy<0     
    azimuth22 = 180-azimuth2    # dx<0 & dy>0
    angle = -(azimuth11-azimuth22)
    return angle

def get_mask_simple(direction_output_tile,direction_output_mask,direction_final, direction_rotated,Tiles,Point_WGS84, name, shooting_radius, sis_coord):
    transformer = Transformer.from_crs(sis_coord['WGS_84'],sis_coord['Pseudo_Mercator'])  
    Data_json = {}
    for num in range(len(Point_WGS84)-1):        
        id_line=num
       
        path = os.path.join(direction_output_tile,'tile_satellite('+name+'_'+str(id_line)+').jpg')
        mTile = Image.open(path, 'r') 
        
        fig = plt.figure(frameon=False,figsize=(mTile.size[0], mTile.size[1]), dpi=1)
        fig.patch.set_facecolor('black') 
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
         
        point0 = transformer.transform(Point_WGS84[num][1],Point_WGS84[num][0])
        point1 = transformer.transform(Point_WGS84[num+1][1],Point_WGS84[num+1][0])

        box = Tiles[(Point_WGS84[num],Point_WGS84[num+1])]['box']
        Data_json[id_line]={'start':Point_WGS84[num],'finish':Point_WGS84[num+1],'polygon:':box}
        minXY,maxXY = box[0],box[2]
        min_point = transformer.transform(minXY[1],minXY[0])
        p1,p3 = transformer.transform(box[1][1],box[1][0]), transformer.transform(box[3][1],box[3][0])
        max_point = transformer.transform(maxXY[1],maxXY[0])
        
        poly = Polygon ([min_point,p1,max_point,p3])
        ax.add_patch(PolygonPatch(poly, fc='black', ec='black', alpha=1, zorder=1))            
        line = LineString([point0,point1])  
        dilated = line.buffer(shooting_radius, cap_style = 2, join_style=2)
#        ax.add_patch(PolygonPatch(dilated, fc='yellow', ec='yellow', alpha=0.7, zorder=1))    
            
        

        poly1 = dilated.difference(poly)       
#        print(poly1.is_empty)
        if poly1.is_empty==False:
#            print(poly1.geom_type)
            max_delta=[]
            if poly1.geom_type=='MultiPolygon':
                for p in poly1:
                    l_points = list(p.exterior.coords)
                    for j in range(len(l_points)-1):
                        max_delta.append(LineString([l_points[j],l_points[j+1]]).length)
            
            new_dist= dist_n(Point_WGS84[num],Point_WGS84[num+1])
            delX = (point0[0]-point1[0])/new_dist
            delY = (point0[1]-point1[1])/new_dist
            if len(poly1)==2:
                delx =min(max_delta)*(delX/2)
                dely =min(max_delta)*(delY/2)
            else:
                delx =np.mean(max_delta)*(delX/2)
                dely =np.mean(max_delta)*(delY/2)
            
            new_point0 = (point0[0]-delx,point0[1]-dely)
            new_point1 = (point1[0]+delx,point1[1]+dely)   
#            print(point0, new_point0)
            
            line_new = LineString([new_point0,new_point1])  
            dilated_new = line_new.buffer(shooting_radius, cap_style = 2, join_style=2)
            ax.add_patch(PolygonPatch(dilated_new, fc='white', ec='white', alpha=1, zorder=1)) 
#            plot_line(ax, line_new ,'red' , w = 100,alpha =1)
#            ax.plot(new_point0[0],new_point0[1],'o', markersize=1000, color='blue')
#            ax.plot(new_point1[0],new_point1[1],'o', markersize=1000, color='blue')
            angle = get_angle(new_point0,new_point1)      
                    
                    
        elif poly1.is_empty:            
            line = LineString([point0,point1])  
            dilated = line.buffer(shooting_radius, cap_style = 2, join_style=2)
            
            ax.add_patch(PolygonPatch(dilated, fc='white', ec='white', alpha=1, zorder=1))    
#            plot_line(ax, line ,'green' , w = 100,alpha =1)   
            angle = get_angle(point0,point1)   
            
        else:
            print(poly1.geom_type)
            angle=0
        
        
#            
        xrange, yrange = [min_point[0], max_point[0]], [min_point[1], max_point[1]]
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        ax.set_aspect(1) 
        find_folder(direction_output_mask)   
        path = os.path.join(direction_output_mask,'buffer_coord('+name+'_'+str(id_line)+').jpg')
        fig.savefig(path)
        plt.close()
        del fig
        direct = direction_output_mask.split('\\')[0]
        with open(direct+'\coordinates.json', 'w') as f:
            json.dump(Data_json, f, ensure_ascii=False, indent=4)
        crop_image(direction_output_tile,direction_output_mask,direction_final,direction_rotated,name,id_line,angle)

def crop_image(direction_output_tile,direction_output_mask,direction_final,direction_rotated,name,id_line,angle):
    try:
        path = os.path.join(direction_output_tile,'tile_satellite('+name+'_'+str(id_line)+').jpg')
        merged_tiles = Image.open(path, 'r')
    except: 
        raise Exception('Ошибка данных, нет тайла:','tile_satellite('+name+'_'+str(id_line)+').jpg')
    map_tile = np.asarray(merged_tiles)
    try:
        path = os.path.join(direction_output_mask,'buffer_coord('+name+'_'+str(id_line)+').jpg')
        buffer = imread(path, cv2.IMREAD_GRAYSCALE)
    except:
        raise Exception('Ошибка данных, нет маски маршрута:','buffer_coord('+name+'_'+str(id_line)+').jpg')
    gray = cv2.cvtColor(buffer, cv2.COLOR_BGR2GRAY)    
    _, mask = cv2.threshold(gray, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    
    tile_x,   tile_y, _ = map_tile.shape
    buffer_x, buffer_y  = mask.shape

    x_buffer = min(tile_x, buffer_x)
    x_half_buffer = mask.shape[0]//2
    buffer_mask = mask[x_half_buffer-x_buffer//2 : x_half_buffer+x_buffer//2+1, :tile_y]
    tile_to_mask = map_tile[x_half_buffer-x_buffer//2 : x_half_buffer+x_buffer//2+1, :tile_y]

    masked = cv2.bitwise_and(tile_to_mask,tile_to_mask,mask = buffer_mask)
    tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(masked)
    rgba = [b,g,r, alpha]
    masked_tr = cv2.merge(rgba,4)
    
    img = Image.fromarray(masked_tr, 'RGBA')
    
    find_folder(direction_final)
    img.save(direction_final+'/final('+name+'_'+str(id_line)+').png')
    rotate_img(direction_final,direction_rotated,name,id_line,angle)
    del img

def rotate_img(direction_final,direction_rotated,name,id_line,angle):
    
    path = os.path.join(direction_final,'final('+name+'_'+str(id_line)+').png')
    final_pict = Image.open(path, 'r')
    map_image = np.asarray(final_pict)
    rotation_matrix = cv2.getRotationMatrix2D((map_image.shape[0] / 2, map_image.shape[1] / 2), angle, 1)
    rotated_image = cv2.warpAffine(map_image, rotation_matrix, (map_image.shape[0],map_image.shape[1]), flags=cv2.INTER_LINEAR)

    find_folder(direction_rotated)
    img = Image.fromarray(rotated_image, 'RGBA')
    img.save(direction_rotated+'/rotated('+name+'_'+str(id_line)+').png')
    del img
    

def function_call(step_length,shooting_radius,binKey,path):
    sis_coord = {'Pseudo_Mercator':3857, 'WGS_84': 4326 }    
    direct = {'in':'input',
              'out_masks': 'masks',
              'out_tiles': 'tile_pict',
              'out_final': 'final',
              'merge_tiles': 'merge_tiles',
              'rotated': 'rotated',}
    
    
    if check_path(path)==True:
        directions,view,Point_WGS84,name = list_of_points(path,direct,step_length,shooting_radius,sis_coord)            
            
        Tiles,Z = get_tile_dict(Point_WGS84,binKey)        
        all_zooms=dict(zip(list(Z.values()),[list(Z.values()).count(i) for i in list(Z.values())]))
        while len(all_zooms.keys())!=1:
                Tiles,Z,Point_WGS84= check_zoom(Tiles,Z,all_zooms,Point_WGS84,binKey,sis_coord)
                all_zooms=dict(zip(list(Z.values()),[list(Z.values()).count(i) for i in list(Z.values())]))
        Tiles = save_tile_as_picture(Tiles,directions['out_tiles'],name)
            
        fig, ax = plt.subplots(num=None, figsize=(24,12),  dpi=30)
        L,ax = path_plan(Tiles, Point_WGS84, ax)
        plt.savefig(str(name)+'/plan ('+str(Point_WGS84[0])+'_'+str(Point_WGS84[-1])+').jpg')
        plt.close()        
        get_mask_simple(directions['out_tiles'],directions['out_masks'], directions['out_final'], directions['rotated'],Tiles,Point_WGS84, name, shooting_radius, sis_coord)
#    else:
#        print('неверно задан маршрут:', path )




def test1_siple_path(step_length,shooting_radius,binKey):
    ###'input/input_data.json' - файл с  примерами маршрутов
    path =[[30.308071, 59.957235], [30.31744, 59.952105]]
    function_call(step_length,shooting_radius,binKey,path)

def test2_multi_directions(binKey):
    step_length,shooting_radius=100,50
    path =[[30.320787, 59.939174], [30.322353, 59.935175], [30.316753, 59.936207], [30.316302, 59.938844], [30.317568, 59.939995], [30.320787, 59.939174]]
#    path =[[30.317568, 59.939995],[30.316302, 59.938844],[30.316753, 59.936207],[30.322353, 59.935175],  [30.320787, 59.939174] ]#, [30.320787, 59.939174]]
    
    function_call(step_length,shooting_radius,binKey,path)

def test3 (step_length,shooting_radius, binKey):
    sis_coord = {'Pseudo_Mercator':3857, 'WGS_84': 4326 } 
    transformer_84_PM = Transformer.from_crs(sis_coord['WGS_84'],sis_coord['Pseudo_Mercator'])  
    transformer_PM_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'],sis_coord['WGS_84'])  
    ###создание маршрута
    start = [39.654624,47.273392]     
    start_PM = transformer_84_PM.transform(start[1],start[0])
    finish_PM = (start_PM[0]-700,start_PM[1]-690)
    finish = transformer_PM_84.transform(finish_PM[0],finish_PM[1])
    ########
    path=[start,[finish[1],finish[0]]]
    function_call(step_length,shooting_radius,binKey,path)
    
def test4 (step_length,shooting_radius, binKey):
    with open(os.path.join('input','input_data.json'), 'r', encoding='utf-8') as j:
        List = json.load(j)
    for path in List:
        function_call(step_length,shooting_radius,binKey,path)
        


if __name__ == '__main__2':
    
    ###НА ВХОДЕ: 1. марштур в виде списка, можно задать 2 типа маршрутов:
    ###                                     *простой - 2 точки [start, finish];
    ###                                     *составной - несколько точек [start, point1, point2,..., finish];
    ###            point = [долгота, широта] - точчи задаются листом;
    ###            'input/input_data.json' - файл с набором разных маршрутов.
    ###
    ###         2. Ключ BingAPI.
    
    ###         3. step_length - Дискретный шаг сьемки (определяяет масштаб получаемого снимка), shooting_radius - радиус сьемки, если маршрут:
    ###                                         *простой(start, finish), оба параметра задаются пользователем;
    ###                                         *составной(start, point1, point2,..., finish), параметры являются дефолтными step_length = 100 метром, 
    ###                                                                                                                      shooting_radius = 50м.
    ###           
    
    ###НА ВЫХОДЕ: 
    ###         КАТАЛОГ c названием, определяющим первую и последную точку маршрута, который содержит след каталоги и файлы:
    ###                                     1.  '/tile_pict' - набор спутниковых снимков, сделанных на каждом шаге пути,
    ###                                     2.  '/masks' - набор вспомогательных картинок, представляющих собой шаги,
    ###                                     3.  '/final' - набор финальных картинок, представляющие собой снимки на каждом шаге, 
    ###                                     4.  'plan(___).jpeg' - схематичное представление полного маршрута,
    
               
    
    
        
    binKey = "AjJhQyVMzBNnY6-64Wt0GpVT_MckgYdZYCP5tSOS4mAkhjY1Pso5FEiGN9nNf4et"   
    step_length,shooting_radius = 300, 100  
    
    ###ТЕСТ для простого маршрута 
    #test1_siple_path(step_length,shooting_radius,binKey)
    


##    
##    
    ###ТЕСТ для составного маршрута из 6 точек
    #test2_multi_directions(binKey)
##    
#    ###ТЕСТ, где задана только точка старта, для нее определяется точка финиша на определенном расстоянии
    #test3 (step_length,shooting_radius, binKey)
    
    test4(step_length,shooting_radius, binKey)

if __name__ == '__main__':
    function_call(1000,100,"Agk8Im5rSKvyKNxRK5r3RDwlqQm11T5XP6fm7mtN37tyEK6Yycj3CINqL3PJrH9M",[[30.306299106247973,60.058562937274395],[30.33502441725433,59.984443080544764]])























   
 
    