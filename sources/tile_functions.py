# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:55:37 2021

@author: user
"""

import json,os,pyproj,urllib.request,cv2,io,math
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib as mlt
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from shapely.geometry import Point,LineString,Polygon
from pyproj import Transformer
import copy,shapely
import shutil

geod = pyproj.Geod(ellps='WGS84')
plt.rcParams['savefig.facecolor'] = 'black'

def plot_line(ax, ob,color , w = 0.5,alpha =1):
    x, y = ob.xy
    ax.plot(x, y, color=color, linewidth=w, alpha =1, solid_capstyle='round', zorder=1)

def find_folder(direction_out):
    if os.path.isdir(direction_out):
        shutil.rmtree(direction_out)
    os.mkdir(direction_out)
    
def new_directions(direct,name):    
    find_folder(name)
    directions={}
    for key in direct:
        os.mkdir(name + '\\'+key)
        directions[key]=name + '\\'+key
    return directions

def formart_point(point):
    return (float("{:.5f}".format(point[0])), float("{:.5f}".format(point[1])))

def valid_lonlat(lon, lat):
    lon %= 360
    if lon >= 180:
        lon -= 360
    lon_lat_point = shapely.geometry.Point(lon, lat)
    lon_lat_bounds = shapely.geometry.Polygon.from_bounds(xmin=-180.0, ymin=-90.0, xmax=180.0, ymax=90.0)
    if lon_lat_bounds.intersects(lon_lat_point):
        return lon, lat

def dist_2points(p0,p1):
    a,b= Point(p0),Point(p1)
    _,_,distance1 = geod.inv(a.x, a.y, b.x, b.y)
    return distance1

def get_list_points(start,finish,delta):
    transformer    = Transformer.from_crs(4326,3857)      
    transformer_84 = Transformer.from_crs(3857,4326)  
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

def get_set_points_of_areas(start,finish, dist_sf):
    step_length=100
    remain = dist_sf%step_length
    if remain!=0:
        new_finish=extend_path(start,finish, step_length, remain, dist_sf)
        finish = (formart_point(new_finish)[0],formart_point(new_finish)[1])
        dist_sf=dist_2points(start,finish)
    del remain,new_finish    
    if dist_sf > 1000: 
        if dist_sf//1000==1:
            if dist_sf//1000>5:
                set_points = get_list_points(start,finish,3) 
            else:
                set_points=[start,finish]
        else:
            delta_points = int(dist_sf//1000)+1
            set_points = get_list_points(start,finish,delta_points)   
    else:
        set_points=[start,finish]
    return set_points

def extend_path(start,finish, step_length,remain,dist):
    transformer    = Transformer.from_crs(4326,3857)     
    transformer_84 = Transformer.from_crs(3857,4326)         
    dist_to_new_point = int(step_length-remain)+1            
    point0 = transformer.transform(start[1],start[0])
    point1 = transformer.transform(finish[1],finish[0])
    detX = (point1[0]-point0[0])/dist
    delY = (point1[1]-point0[1])/dist            
    new_finish = (point1[0]+(dist_to_new_point*detX),point1[1]+(dist_to_new_point*delY))
    n_finish = transformer_84.transform(new_finish[0],new_finish[1])
    return (n_finish[1],n_finish[0])

def extend_area(start,finish):
    step_length=300
    transformer    = Transformer.from_crs(4326,3857)
    transformer_84 = Transformer.from_crs(3857,4326)    
    PM_start  = transformer.transform(start[1],start[0])
    PM_finish = transformer.transform(finish[1],finish[0])     
    X_max,X_min = max([PM_start[0],PM_finish[0]]),min([PM_start[0],PM_finish[0]])
    Y_max,Y_min = max([PM_start[1],PM_finish[1]]),min([PM_start[1],PM_finish[1]])
    minXY1 = (X_min-step_length,Y_min-step_length)
    maxXY1 = (X_max+step_length,Y_max+step_length)
    WGS_start1  = transformer_84.transform(minXY1[0],minXY1[1])
    WGS_finish1 = transformer_84.transform(maxXY1[0],maxXY1[1])
    return formart_point((WGS_start1[1],WGS_start1[0])),formart_point((WGS_finish1[1],WGS_finish1[0]))


def create_grid(s_point,f_point,step_length=200):    
    start,finish = extend_area(s_point,f_point)  
    transformer    = Transformer.from_crs(4326,3857)     
    transformer_84 = Transformer.from_crs(3857,4326)   
    point0,point1 = transformer.transform(start[1],start[0]),transformer.transform(finish[1],finish[0])
    minX, maxX = min(point0[0],point1[0]),max(point0[0],point1[0])
    minY, maxY = min(point0[1],point1[1]),max(point0[1],point1[1])
    nx,ny = math.ceil((abs(maxX - minX)/step_length)),math.ceil((abs(maxY - minY)/step_length))
    G,Point_WGS84={},[]
    for i in range(nx+1):
        G[i]=[]
        for j in range(ny+1):
            xy = (minX+ (i*step_length),minY+(j*step_length))
            xy_84= transformer_84.transform(xy[0],xy[1])   
            G[i].append((xy_84[1],xy_84[0]))               
    for key in G.keys():
        if key%2==0:
            Point_WGS84.extend(G[key])
        else:
            Point_WGS84.extend(G[key][::-1])   
    return G,Point_WGS84,start,finish

def makeRequest(rq):
    f = urllib.request.urlopen(rq)
    data = f.read()
    return data

def get_bing_tile(point1,point2,binKey,type_of_imagery,size_of_image=(1000,750)):
    lat_max,lat_min = max([point1[1],point2[1]]),min([point1[1],point2[1]])
    lon_max,lon_min = max([point1[0],point2[0]]),min([point1[0],point2[0]])
    rq = "http://dev.virtualearth.net/REST/V1/Imagery/Map/{0}?mapArea={1},{2},{3},{4}&zoomLevel=19&ms={5},{6}&mmd={7}&key={8}"
    rq = "http://dev.virtualearth.net/REST/V1/Imagery/Map/{0}?mapArea={1},{2},{3},{4}&zoomLevel=19&ms={5},{6}&mmd={7}&key={8}"
    rq1 = rq.format(type_of_imagery,lat_min, lon_min, lat_max, lon_max, size_of_image[0], size_of_image[1], 0, binKey)
    rq2 = rq.format(type_of_imagery,lat_min, lon_min, lat_max, lon_max, size_of_image[0], size_of_image[1], 1, binKey)
    
    tile = makeRequest(rq1)
    retJson = makeRequest(rq2).decode('utf-8')
    obj = json.loads(retJson)
    zoom = int(obj['resourceSets'][0]['resources'][0]['zoom'])
    coords = obj['resourceSets'][0]['resources'][0]['bbox']
    size_picture = (int(obj['resourceSets'][0]['resources'][0]['imageWidth']),int(obj['resourceSets'][0]['resources'][0]['imageHeight']))
    return tile,zoom,[coords[1],coords[3],coords[0],coords[2]],size_picture

def get_tile_dict(Point_WGS84,G,binKey,type_of_imagery,name,direction_output):
    Tiles={}
    for i in range(len(Point_WGS84)-1):      
        for g in G.keys():
            if  Point_WGS84[i] in G[g] and Point_WGS84[i+1] in G[g]:
                if Point_WGS84[i][1]==Point_WGS84[i+1][1]:
                    point1,point2 = Point_WGS84[i],(Point_WGS84[i+1][0],Point_WGS84[i+1][1]+0.000000001)
                else:
                    point1,point2 = Point_WGS84[i],Point_WGS84[i+1]
                tile,zoom,coords,size_picture = get_bing_tile(point1,point2,binKey,type_of_imagery)
                Tiles[(Point_WGS84[i],Point_WGS84[i+1])]={'box': [(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2])],
                                                          'size': size_picture,
                                                          'del_width': dist_2points((coords[0],coords[2]),(coords[1],coords[2]))/size_picture[0],
                                                          'del_high':  dist_2points((coords[0],coords[2]),(coords[0],coords[3]))/size_picture[1], 
                                                          'polygon':Polygon([(coords[0],coords[2]),(coords[0],coords[3]),(coords[1],coords[3]),(coords[1],coords[2]),(coords[0],coords[2])]),
                                                          'line':   LineString([Point_WGS84[i],Point_WGS84[i+1]]),
                                                          'start':  Point_WGS84[i],
                                                          'finish': Point_WGS84[i+1],
                                                          'zoom': zoom,
                                                          'id': i,
                                                          'name_pict': name+'_'+str(i)}        
                image = Image.open(BytesIO(tile))
                try:
                    path_tile = os.path.join(direction_output,'tile_'+name+'_'+str(i)+'.jpg')
                    image.save(path_tile)
                except: 
                    path_tile = os.path.join(direction_output,'tile_'+name+'_'+str(i)+'.png')
                    image.save(path_tile)
                image.close()
    return Tiles

def get_parametrs(start,finish,first_tile,Tiles,L):
    if start[0]<finish[0]: #х увеличивается 
        width = 0 # ширина рисунка начинается с 0  
        if start[1]<finish[1]: #y увеличивается 
            high = Tiles[first_tile]['size'][1]*(len(L)-1) # высота рисунка начинается с последнего тайла            
        else:
            high = 0
    elif start[0]>finish[0]:
        width=Tiles[L[0]]['size'][0]*(len(L)-1) # ширина рисунка начинается с последнего тайла 
        if start[1]<finish[1]: #y увеличивается 
            high = Tiles[first_tile]['size'][1]*(len(L)-1) # высота рисунка начинается с последнего тайла            
        else:
            high = 0 # высота рисунка начинается с 0
    else:
        width = 0 # ширина рисунка начинается с 0  
        if start[1]<finish[1]: #y увеличивается 
            high = Tiles[first_tile]['size'][1]*(len(L)-1) # высота рисунка начинается с последнего тайла            
        else:
            high = 0
    return width,high

def merge_pict(Tiles,Point_WGS84):
    L = list(Tiles.keys())
    first_tile=L[0]
    width,high  = get_parametrs(Tiles[L[0]]['start'],Tiles[L[0]]['finish'],L[0],Tiles,L)
    W,H=[width],[high]
    C={first_tile:(width,high)}        
    for i in range(1,len(L)):
        key,priv_key= L[i],L[i-1]        
        intersect_poly =Tiles[priv_key]['polygon'].intersection(Tiles[key]['polygon'])
        bounds = intersect_poly.bounds            
        for idy in [1,3]:
            point=(bounds[0],bounds[idy])
            if Point(point).touches(Tiles[priv_key]['polygon']) and Point(point).touches(Tiles[key]['polygon']):
                intersect_point = point
                idY=idy    
        if Tiles[priv_key]['box'][0][0]-Tiles[key]['box'][0][0]>0: # клетка находится левеее(х уменьшается)
            if idY==3:
                id_box= 1
            else: 
                id_box= 0
            del_width = dist_2points(intersect_point,Tiles[key]['box'][id_box]) 
            width = C[priv_key][0]-round(del_width/Tiles[key]['del_width'])                 
            if Tiles[priv_key]['box'][0][1]-Tiles[key]['box'][0][1]>0: # клетка находится ниже  (у уменьшается)
                del_high = dist_2points(intersect_point,Tiles[priv_key]['box'][id_box]) 
                high = C[priv_key][1]+round(del_high/Tiles[priv_key]['del_high'])
            else: # клетка находится выше  (у увеличивается)
                del_high = dist_2points(intersect_point,Tiles[priv_key]['box'][id_box]) 
                high = C[priv_key][1]-round(del_high/Tiles[priv_key]['del_high'])                
        elif Tiles[priv_key]['box'][0][0]-Tiles[key]['box'][0][0]<0:       # клетка находится правее(х увеличиваетсся)    
            if idY==3:
                id_box= 1
            else: 
                id_box= 0
            del_width = dist_2points(intersect_point,Tiles[priv_key]['box'][id_box]) 
            width = C[priv_key][0]+round(del_width/Tiles[priv_key]['del_width'])                 
            if Tiles[priv_key]['box'][0][1]-Tiles[key]['box'][0][1]>0: # клетка находится ниже  (у уменьшается)
                del_high = dist_2points(intersect_point,Tiles[key]['box'][id_box]) 
                high = C[priv_key][1]+round(del_high/Tiles[key]['del_high'])                
            else: # клетка находится выше  (у увеличивается)                    
                del_high = dist_2points(intersect_point,Tiles[key]['box'][id_box]) 
                high = C[priv_key][1]-round(del_high/Tiles[key]['del_high'])
        else:
            width = C[priv_key][0]
            p0,p1=(bounds[0],bounds[1]),(bounds[0],bounds[3])
            del_high = dist_2points(p0,p1) 
            if Tiles[priv_key]['box'][0][1]-Tiles[key]['box'][0][1]>0: # клетка находится ниже  (у уменьшается)
                high = C[priv_key][1]+Tiles[priv_key]['size'][1]-round(del_high/Tiles[priv_key]['del_high']) 
            else: 
                high = C[priv_key][1]-Tiles[priv_key]['size'][1]+round(del_high/Tiles[priv_key]['del_high'])                 
        
        C[key]=(width,high)
        W.append(width)
        H.append(high)
    sizeW,sizeH,C_new = size_pict(W,H,C)
    return sizeW,sizeH,C_new

def size_pict(W,H,C):   
    C_new = copy.deepcopy(C)
    deltaY,deltaX=min(H),min(W)
    cofY=-1
    cofX=-1    
    for c in C_new.keys():
        C_new[c]=(C[c][0]+cofX*deltaX,C[c][1]+cofY*deltaY)
    sizeH=max(H)+cofY*deltaY
    sizeW=max(W)+cofX*deltaX
    return sizeW,sizeH,C_new

def final_pict(sizeW,sizeH,C_new,Tiles,directions,name):   
    Merged_tiles = {}   
    X1,Y1=[],[]
    for c in C_new.keys(): 
        X1.extend(list(zip(*Tiles[c]['box']))[0])
        Y1.extend(list(zip(*Tiles[c]['box']))[1])
    Merged_tiles[name] = {'id_tiles': list(C_new.keys()),
                                 'min_crop_box': (min(X1),min(Y1)),
                                 'max_crop_box': (max(X1),max(Y1)),
                                 'X_full_box': (min(X1),max(X1)),
                                 'Y_full_box': (min(Y1),max(Y1))}
    
    L = list(Tiles.keys())
    size_pictures = Tiles[L[0]]['size']        
    new_image = Image.new('RGBA',(sizeW+size_pictures[0],sizeH+size_pictures[1]), (250,250,250)) 
    for key in Merged_tiles[name]['id_tiles']:
        name_tile = Tiles[key]['name_pict']
        try:
            path = os.path.join(directions['tiles'],'tile_'+name_tile+'.jpg')
            image =Image.open(path, 'r')
        except:
            path = os.path.join(directions['tiles'],'tile_'+name_tile+'.png')
            image =Image.open(path, 'r')
        new_image.paste(image,C_new[key])
    rgb_im = new_image.convert('RGBA') 
    path1 = os.path.join(directions['merged_tiles'],name+'.png')
    rgb_im.save(path1)
    del new_image ,rgb_im  
        
    return Merged_tiles
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


def mask(point1,w,h,angle,area_polygon):    
    poly = [np.array([point1[0]-2*w,point1[1]-2*h]),np.array([point1[0]-2*w,point1[1]+2*h]),np.array([point1[0]+2*w,point1[1]+2*h]),np.array([point1[0]+2*w,point1[1]-2*h])]  
    fig, ax = plt.subplots(num=None, figsize=(24,12),  dpi=30)
    rect2 = mlt.patches.Polygon(np.array(poly),fc='yellow',alpha=0.3)   
    td2dis = ax.transData
    ax.add_patch(rect2)
    tr = mlt.transforms.Affine2D().rotate_deg_around(point1[0], point1[1],-angle)
    t3 = mlt.patches.Polygon(rect2.get_xy(),fc='red',alpha=0.8)
    t3.set_transform(tr + td2dis)
    ax.add_patch(t3)
    coords = t3.get_patch_transform().transform(t3.get_path().vertices[:-1])
    new_poly = Polygon(tr.transform(coords))       
    inter=Polygon(new_poly).difference(area_polygon)
    plt.clf()
    plt.cla()
    plt.close('all')
    del fig
    return inter,tr,coords

def size_of_mask(maxXY,minXY,Points,angle,area_polygon):
    transformer    = Transformer.from_crs(4326,3857)
    x,y=[],[]
    for p in [Points[0],Points[-1]]: 
        x.extend([dist_2points(p,(p[0],maxXY[1])),dist_2points(p,(p[0],minXY[1]))])    
        y.extend([dist_2points(p,(maxXY[0],p[1])),dist_2points(p,(minXY[0],p[1]))])        
    min_dist_to_axisX,min_dist_to_axisY = min(x),min(y)
    del x,y,p
    m1,m2=dist_2points(minXY,(minXY[0],maxXY[1])),dist_2points(minXY,(maxXY[0],minXY[1]))
    if m1<m2:
        if min_dist_to_axisY>min_dist_to_axisX:
            w = int(min_dist_to_axisX)
            h = int(4*(min_dist_to_axisX//3))
        else:
            h = int(min_dist_to_axisY)
            w = int(3*(min_dist_to_axisY//4))
    else:
        if min_dist_to_axisY>min_dist_to_axisX:
            h = int(min_dist_to_axisX)
            w = int(4*(min_dist_to_axisX//3))
        else:
            w = int(min_dist_to_axisY)
            h = int(3*(min_dist_to_axisY//4))
    for p in [Points[0],Points[-1]]:
        point1 = transformer.transform(p[1],p[0])
        inter,tr,coords = mask(point1,w,h,angle,area_polygon)
        while inter.is_empty==False:
            D=[]
            if inter.geom_type == "Polygon" or inter.geom_type == "MultiPolygon":
                for c in tr.transform(coords):
                    line0 =LineString([point1,c])
                    if line0.intersects(inter):
                        ll = line0.intersection(inter)
                        d = ll.length/2
                        D.append(d)
            if len(D)!=0:
                diff = max(D)
                if m1<m2:  
                    if min_dist_to_axisY>min_dist_to_axisX:
                        w = int(w-diff/1.7)
                        h = int(4*(w//3))
                    else:
                        h = int(h-diff/1.7)
                        w = int(4*(h//3))
                else:
                    if min_dist_to_axisY>=min_dist_to_axisX:
                        h= int(h-diff/1.7)
                        w = int(4*(h//3))
                    else:
                        w = int(w-diff/1.7)
                        h= int(4*(w//3))
            inter,tr,coords = mask(point1,w,h,angle,area_polygon)
    return w,h

def get_img_from_fig(fig, dpi=1):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    del fig,img_arr
    return img 

def crop(buffer, map_tile, direction_no_rotated, p1): 
    gray = cv2.cvtColor(buffer, cv2.COLOR_BGR2GRAY)    
    _, mask = cv2.threshold(gray, thresh=180, maxval=255, type=cv2.THRESH_BINARY)    
    tile_x,   tile_y, _ = map_tile.shape
    buffer_x, buffer_y  = mask.shape
    x_buffer = min(tile_x, buffer_x)
    x_half_buffer = mask.shape[0]//2
    
    buffer_mask  = mask[x_half_buffer-x_buffer//2 : x_half_buffer+x_buffer//2+1, :tile_y]
    tile_to_mask = map_tile[x_half_buffer-x_buffer//2 : x_half_buffer+x_buffer//2+1, :tile_y]
    masked = cv2.bitwise_and(tile_to_mask,tile_to_mask,mask = buffer_mask)
    tmp = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r, a = cv2.split(masked)
    rgba = [b,g,r, alpha]
    masked_tr = cv2.merge(rgba,4)
    
    try:
        gray_img= cv2.cvtColor(masked_tr,cv2.COLOR_BGR2GRAY)
        edges= cv2.Canny(gray_img, 50, 255)
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        x,y=[],[]
        for contour_line in contours:
            for contour in contour_line:
                x.append(contour[0][0])
                y.append(contour[0][1])
        x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
        del x,y
        cropped = masked_tr[y1:y2, x1:x2]
        img2 = Image.fromarray(cropped, 'RGBA')
        img2.save(direction_no_rotated+'/'+str(p1[1])+'_'+str(p1[0])+'.png')
        img2.close()

        full_box = map_tile [y1:y2, x1:x2]
        img3 = Image.fromarray(full_box, 'RGBA')
        img3.save(direction_no_rotated+'/box_'+str(p1[1])+'_'+str(p1[0])+'.png')
        img3.close()
    except:
        img2 = Image.fromarray(masked_tr, 'RGBA')
        img2.save(direction_no_rotated+'/'+str(p1[1])+'_'+str(p1[0])+'.png')
        img2.close()
        
    del gray,mask,buffer_mask,tile_to_mask,masked,tmp,masked_tr,gray_img,edges,contours, hierarchy,
    cv2.waitKey(0) 
    cv2.destroyAllWindows()

def no_rotated_image(fig,ax, p1,angle,w,h,map_tile,min_point,max_point,directions):
    transformer    = Transformer.from_crs(4326,3857)
    plt.ioff()
    AP = Polygon([min_point,(min_point[0],max_point[1]),max_point,(max_point[0],min_point[1])])
    ax.add_patch(PolygonPatch(AP, fc='black', ec='black', alpha=1, zorder=1))
    
    point1 = transformer.transform(p1[1],p1[0])
    poly = [np.array([point1[0]-w,point1[1]-h]),
                np.array([point1[0]-w,point1[1]+h]),
                np.array([point1[0]+w,point1[1]+h]),
                np.array([point1[0]+w,point1[1]-h])]
    rect2 = mlt.patches.Polygon(np.array(poly),fc='yellow',alpha=0.3,zorder=99)   
    td2dis = ax.transData
    tr = mlt.transforms.Affine2D().rotate_deg_around(point1[0], point1[1],-angle)
    t3=mlt.patches.Polygon(rect2.get_xy(),fc='white',alpha=1,zorder=1)
    t3.set_transform(tr + td2dis)
        
    coords = t3.get_patch_transform().transform(t3.get_path().vertices[:-1])
    new_coords = list(tr.transform(coords))
    ax.add_patch(PolygonPatch(Polygon(new_coords), fc='white', ec='white', alpha=1, zorder=1))
#    direction_output_mask = directions['masks']   
#    path_buf = os.path.join(direction_output_mask,'mask_'+str(p1[1])+'_'+str(p1[0])+'.jpg')
#    plt.savefig(path_buf, transparent=False,facecolor= 'black')
    buffer = get_img_from_fig(fig)
    
    del rect2,td2dis,tr,t3,coords
    crop(buffer,map_tile,directions['not_rotated_img'],p1 )
    del buffer
    return new_coords
    
def rotate_img(angle,direction_final,direction_rotated,p1):    
    path_final = os.path.join(direction_final,str(p1[1])+'_'+str(p1[0])+'.png')
    final_img = Image.open(path_final, 'r')
    map_image = np.asarray(final_img)
    del final_img
    rotation_matrix = cv2.getRotationMatrix2D((map_image.shape[0] / 2 , map_image.shape[1] / 2 ), angle, 1)
    rotated_image = cv2.warpAffine(map_image, rotation_matrix, (map_image.shape[0],map_image.shape[1]), flags=cv2.INTER_LINEAR)    
    gray_img= cv2.cvtColor(rotated_image,cv2.COLOR_BGR2GRAY)
    edges= cv2.Canny(gray_img, 60, 255)
    contours,_= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours)!=0:
        x,y=[],[]
        for contour_line in contours:
            for contour in contour_line:
                x.append(contour[0][0])
                y.append(contour[0][1])
        x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
        cropped = rotated_image[y1:y2, x1:x2]
        img2 = Image.fromarray(cropped, 'RGBA')
        img2.save(direction_rotated+'/'+str(p1[1])+'_'+str(p1[0])+'.png')
    else:
        img2 = Image.fromarray(map_image, 'RGBA')
        img2.save(direction_rotated+'/'+str(p1[1])+'_'+str(p1[0])+'.png')
    img2.close()
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    del img2,rotation_matrix,rotated_image,gray_img,contours,contour_line,x,y,cropped

def call_tiles_functions(binKey,set_points,type_img,directions,name):
    Type_of_imagery = {'Satellite': 'Aerial',
                       'SatelliteLabels': 'AerialWithLabels',
                       'Road'     : 'Road',
                       'DarkRoad' : 'CanvasDark',
                       'LightRoad': 'CanvasLight',
                       'GrayRoad' : 'CanvasGray'}    
    Merged_tiles = {}
    for i in range(len(set_points)-1):   
        s_point,f_point=set_points[i],set_points[i+1]
        G,Point_WGS84,new_s,new_f = create_grid(s_point,f_point)
        tiles_name = str(s_point)+'_'+str(f_point)
        Tiles = get_tile_dict(Point_WGS84,G,binKey,Type_of_imagery[type_img],tiles_name,directions['tiles'])
        sizeW,sizeH,C_new = merge_pict(Tiles,Point_WGS84)
        M = final_pict(sizeW,sizeH,C_new,Tiles,directions,tiles_name)
        Merged_tiles.update(M)
    with open(name+'\\Merged_tiles.json', 'w') as outfile:
        json.dump(Merged_tiles, outfile, ensure_ascii=False, indent=4)  
    return Merged_tiles

def get_all_pict(set_points,Merged_tiles,directions,shooting_step,name):
    transformer = Transformer.from_crs(4326,3857)
    transformer_84 = Transformer.from_crs(3857,4326)  
    Data_json={}
    for i in range(len(set_points)-1):
        s_point,f_point=set_points[i],set_points[i+1]
        tiles_name = str(s_point)+'_'+str(f_point)
        dist = dist_2points(s_point,f_point)
        if dist%shooting_step==0:
            delta = int(dist//shooting_step)
        else:
            delta = int(dist//shooting_step)+1
        Points = get_list_points(s_point,f_point,delta)
        
        path_merged = os.path.join(directions['merged_tiles'],tiles_name+'.png')
        mTile = Image.open(path_merged, 'r') 
        map_tile = np.asarray(mTile)
        size_image = (map_tile.shape[1],map_tile.shape[0])
        mTile.close()
        del delta,dist,path_merged,mTile
        minXY,maxXY = Merged_tiles[tiles_name]['min_crop_box'], Merged_tiles[tiles_name]['max_crop_box']       
        min_point,max_point = transformer.transform(minXY[1],minXY[0]),transformer.transform(maxXY[1],maxXY[0])    
        p1,p3 = (min_point[0],max_point[1]),(max_point[0],min_point[1])
        area_polygon = Polygon([min_point,p1,max_point,p3])
        
        angle = get_angle(s_point,f_point)  
        w,h = size_of_mask(maxXY,minXY,Points,angle,area_polygon)
        plt.ioff()
        fig= plt.figure(figsize=(size_image[0], size_image[1]), dpi=1)
        fig.patch.set_facecolor('black')
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.patch.set_facecolor('black')
        fig.add_axes(ax) 
        ax.set_aspect(1)
        xrange, yrange = [min_point[0], max_point[0]], [min_point[1], max_point[1]]
        ax.set_xlim(*xrange)
        ax.set_ylim(*yrange)
        for j in range(len(Points)):
            Data_json [str(i)+'_'+str(j)] = {'center_point':(Points[j][1],Points[j][0]), 'polygon_coordinates': [], 'box_coordinates':[]}
            coords = no_rotated_image(fig,ax, Points[j],angle,2*w,2*h,map_tile,min_point,max_point,directions)
            x,y=[],[]
            for coord in coords:
                point=transformer_84.transform(coord[0],coord[1])
                Data_json[str(i)+'_'+str(j)]['polygon_coordinates'].append(point) 
                x.append(point[0])
                y.append(point[1])
            rotate_img(angle,directions['not_rotated_img'],directions['rotated_img'],Points[j])
            min_X,min_Y,max_X,max_Y = min(x),min(y),max(x),max(y)            
            Data_json [str(i)+'_'+str(j)]['box_coordinates'] = [(min_X,min_Y),(min_X,max_Y),(max_X,max_Y),(max_X,min_Y)] 
        plt.clf()
        plt.cla()
        plt.close('all')      
    del ax,fig,coords
    with open(name+'/output_data.json', 'w') as f:
        json.dump(Data_json, f,ensure_ascii=False, indent=4)

def function_call (type_img,path,shooting_step,binKey):
    direct = ['tiles','merged_tiles','not_rotated_img','rotated_img', 'masks']
    start   = (formart_point(path[0])[1],formart_point(path[0])[0])
    finish  = (formart_point(path[1])[1],formart_point(path[1])[0])
    dist_sf = dist_2points(start,finish)
    if dist_sf<shooting_step:
        raise ValueError("not valid parameter 'step_length', shooting distance=",dist_sf)
    name = type_img+'_'+str(path[0])+'_'+str(path[1])
    set_points = get_set_points_of_areas(start,finish, dist_sf)
    
    if os.path.isdir(name):
        if os.path.isdir(name + '\\merged_tiles') and os.listdir(name+ '\\merged_tiles')!=[] and os.path.isfile(name+'\\Merged_tiles.json')==True:         
            directions={}
            for key in direct:
                if os.path.isdir(name + '\\'+key)==False:
                    os.mkdir(name + '\\'+key)
                elif key in ['not_rotated_img','rotated_img', 'masks']:
                     shutil.rmtree(name + '\\'+key)
                     os.mkdir(name + '\\'+key)
                directions[key]=name + '\\'+key
            
            with open(name+'\\Merged_tiles.json') as json_file:
                Merged_tiles = json.load(json_file)
            get_all_pict(set_points,Merged_tiles,directions,shooting_step,name)
        else:
            directions=new_directions(direct,name)
            Merged_tiles = call_tiles_functions(binKey,set_points,type_img,directions,name)
            get_all_pict(set_points,Merged_tiles,directions,shooting_step,name)
    else:
        directions=new_directions(direct,name)
        Merged_tiles = call_tiles_functions(binKey,set_points,type_img,directions,name)
        get_all_pict(set_points,Merged_tiles,directions,shooting_step,name)

    

def test1(type_img,shooting_step,binKey):
    path =[[40.754399, -73.98669],[40.769008, -73.97391]]
    function_call(type_img,path,shooting_step,binKey)

def test2(type_img,shooting_step,binKey):    
    path =[[38.708838, -9.131419],[38.712479, -9.139875]]
    function_call(type_img,path,shooting_step,binKey)

def test3(type_img,shooting_step,binKey): 
    sis_coord = {'Pseudo_Mercator':3857, 'WGS_84': 4326 } 
    transformer_84_PM = Transformer.from_crs(sis_coord['WGS_84'],sis_coord['Pseudo_Mercator'])  
    transformer_PM_84 = Transformer.from_crs(sis_coord['Pseudo_Mercator'],sis_coord['WGS_84'])  
    ###создание маршрута
    start = [47.273392,39.654624]  
    start_PM = transformer_84_PM.transform(start[0],start[1])
    finish_PM = [start_PM[0]-690, start_PM[1]-700]
    finish = transformer_PM_84.transform(finish_PM[0],finish_PM[1])   
    ########
    path=[start,finish]
    function_call(type_img,path,shooting_step,binKey)

def test4(type_img,shooting_step,binKey):
    path =[[59.949893,30.314885], [59.951881,30.308791]]
    function_call(type_img,path,shooting_step,binKey)
    
def test5(type_img,shooting_step,binKey):
    path =[[38.708838, -9.131419],[38.712479, -9.139875]]
    function_call(type_img,path,shooting_step,binKey)
    
def test6(type_img,shooting_step,binKey):
    path =[[59.968312, 30.208302],[59.964594, 30.213900]]
    function_call(type_img,path,shooting_step,binKey)






if __name__ == '__main__':
    
    ###НА ВХОДЕ: 1. type_img - тип скачиваемых тайлов type_of_imagery
    ###          2. маршрут в виде списка 2 точки [start, finish];
    ###             point = [широта,долгота] - точки задаются листом ;
    ###          3. crop_step - Дискретный шаг для вырезания картинки, задается в метрах,
    ###          4. Ключ BingAPI.
    ###                                        
    
    ###НА ВЫХОДЕ: 
    ###         КАТАЛОГ c названием, определяющим тип карты, первую и последнюю точку маршрута, который содержит след каталоги и файлы:
	###						1. '/output' - каталог с вспомогательными картинками, которые получаются походу работы кода
	###									* '\tiles' - скачанные тайлы
	###									* '\masks' - созданные маски для вырезания кадров
	###									* '\not_rotated' - вырезанные кадры, не повернутые 
	###						
	###						2. '/final' - каталог, где содержатся финальные картинки
	###									* '\merge_tiles' - объединенные тайлы, если расстояние между точками start, finish > 1000, разбиваются на несколько наборов, чтобы картинки были приемлемого размера
	###									* '\merging_of_tiles' - картинки с постепенным объединением тайлов
	###									* '\rotated_pict' - набор финальных вертикальных картинок 
	###						3.  'coordinates.json' - данные о финальных картинках, старт, финиш, левая нижняя и правая верхняя точки вырезанного кадра   
    ###                     4.  'plan(___).jpeg' - схематичное представление всей съемки
   

    binKey = "AjJhQyVMzBNnY6-64Wt0GpVT_MckgYdZYCP5tSOS4mAkhjY1Pso5FEiGN9nNf4et"  
    Type_of_tile = ['Satellite','SatelliteLabels','Road','DarkRoad','LightRoad','GrayRoad']  
  
    shooting_step = 100    
    #test1('Satellite',shooting_step,binKey)
    test2('Satellite',shooting_step,binKey)
    #test3('Satellite',shooting_step,binKey)
    #test4('Satellite',shooting_step,binKey)
    #test5('Satellite',shooting_step,binKey)
    #test5('Satellite',shooting_step,binKey)
    

    #test1('Road',shooting_step,binKey)
    #test2('DarkRoad',shooting_step,binKey)
    #test3('LightRoad',shooting_step,binKey)
    #test4('GrayRoad',shooting_step,binKey)

