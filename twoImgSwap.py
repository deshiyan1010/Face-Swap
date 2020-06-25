import cv2
import numpy as np
import dlib
import warnings
warnings.filterwarnings("ignore")




def seamless(img,orig,mask,rect):

    x,y,w,h=rect
    centre = (int((x + x + w) / 2), int((y + y + h) / 2))
    seamless_clone = cv2.seamlessClone(img,orig,mask,centre,cv2.NORMAL_CLONE)

    return seamless_clone

def final_overlay(img1,img2):

    rows,cols,channels = img2.shape
    roi = img1

    img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)
    out_img = cv2.add(img1_bg,img2_fg)
    img1 = out_img

    return img1,mask_inv

def overlay(img1,img2,x,y,w=0,h=0,disp=False):

    # h,w,_ = img2.shape
    # roi = img1[y:y+h,x:x+w]
    # img2gray = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    
    
    # mask_inv = cv2.bitwise_not(img2gray) #ORIG
    # mask = cv2.bitwise_not(mask_inv)

    # img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv) 
    # img2_fg = cv2.bitwise_and(img2,img2,mask = mask)

    # # ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY_INV) # EXP
    # # mask_inv = cv2.bitwise_not(mask)

    # # img1_bg = cv2.bitwise_and(roi,roi,mask = mask)
    # # img2_fg = cv2.bitwise_and(img2,img2,mask = mask_inv)

    # out_img = cv2.add(img1_bg,img2_fg)

    # img1[y:y+h,x:x+w] = out_img

    
    # return img1
    #img1 = cv2.GaussianBlur(img1,(3,3),0)
    
    img2_new_face_rect_area = img1[y: y + h, x: x + w]
    img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
    
    mask_triangles_designed_gauss = cv2.GaussianBlur(mask_triangles_designed,(3,3),0)
    mask_triangles_designed_added = cv2.addWeighted(mask_triangles_designed_gauss,2.5,mask_triangles_designed,-0.5,0)
    
    img2 = cv2.bitwise_and(img2, img2, mask=mask_triangles_designed_added)

    img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, img2)
    img1[y: y + h, x: x + w] = img2_new_face_rect_area

    return img1

def show(img):
    cv2.imshow("img",img)
    cv2.waitKey(0)

def create_dict(lst):
    val2ind = {}
    ind2val = {}
    for i,x in enumerate(lst):
        val2ind[x]=i
        ind2val[i]=x

    return val2ind, ind2val

def triangle(num):

    indices = tri_ind2val[num]

    #print(indices)

    pt1_1 = landmark_ind2val[indices[0]]
    pt1_2 = landmark_ind2val[indices[1]]
    pt1_3 = landmark_ind2val[indices[2]]
    pt1 = [pt1_1,pt1_2,pt1_3]

    pt2_1 = landmark_ind2val2[indices[0]]
    pt2_2 = landmark_ind2val2[indices[1]]
    pt2_3 = landmark_ind2val2[indices[2]]       
    pt2 = [pt2_1,pt2_2,pt2_3]

    # cv2.line(frame,pt1_1,pt1_2,(0,0,255),2)
    # cv2.line(frame,pt1_2,pt1_3,(0,0,255),2)
    # cv2.line(frame,pt1_3,pt1_1,(0,0,255),2)

    # cv2.line(frame2,pt2_1,pt2_2,(0,0,255),2)
    # cv2.line(frame2,pt2_2,pt2_3,(0,0,255),2)
    # cv2.line(frame2,pt2_3,pt2_1,(0,0,255),2)

    x1,y1,w1,h1 = cv2.boundingRect(np.array(pt1,dtype=np.int32))
    x2,y2,w2,h2 = cv2.boundingRect(np.array(pt2,dtype=np.int32))

    pt_lst_1_aff = [(pt1_1[0]-x1,pt1_1[1]-y1),(pt1_2[0]-x1,pt1_2[1]-y1),(pt1_3[0]-x1,pt1_3[1]-y1)]
    pt_lst_2_aff = [(pt2_1[0]-x2,pt2_1[1]-y2),(pt2_2[0]-x2,pt2_2[1]-y2),(pt2_3[0]-x2,pt2_3[1]-y2)]

    return pt1,pt2,pt_lst_1_aff,pt_lst_2_aff

def crop(img,points):
    
    x,y,w,h = cv2.boundingRect(np.array(points, dtype=np.int32))   #ORIG
    cropped = img[y:y+h,x:x+w,:]

    mask = np.zeros_like(img)
    roi_corners = np.array([points], dtype=np.int32)
    cv2.fillPoly(mask, roi_corners,255)
    masked_image = cv2.bitwise_and(img,img,mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY))
    #show(masked_image)
    return np.array(masked_image[y:y+h,x:x+w,:]),w,h,x,y

    # x,y,w,h = cv2.boundingRect(np.array(points, dtype=np.int32))
    # cropped_mask = np.zeros((h,w),np.uint8)
    # cv2.fillConvexPoly(cropped_mask, aff_points, 255)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

frame = cv2.imread("img_vid/bradley_cooper.jpg",-1)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

for face in faces:

    landmarks = predictor(gray, face)
    landmark_lst = []
    for n in range(0, 68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmark_lst.append((x,y))
        #cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    landmark_lst_32 = np.array(landmark_lst,dtype=np.int32)
    landmark_val2ind,landmark_ind2val = create_dict(landmark_lst)
    convexHull = cv2.convexHull(landmark_lst_32)
    rect_1 = cv2.boundingRect(convexHull)

    subdiv = cv2.Subdiv2D(rect_1)
    subdiv.insert(landmark_lst)
    triangles = subdiv.getTriangleList()

    triangle_lst = []

    for t in triangles:
        pt1 = (t[0],t[1])
        pt2 = (t[2],t[3])
        pt3 = (t[4],t[5])

        triangle_lst.append((pt1,pt2,pt3))

        # cv2.line(frame,pt1,pt2,(0,0,255),2)
        # cv2.line(frame,pt2,pt3,(0,0,255),2)
        # cv2.line(frame,pt3,pt1,(0,0,255),2)
        
    vert2ind = []

    for pts in triangle_lst:
        x = landmark_val2ind[pts[0]]
        y = landmark_val2ind[pts[1]]
        z = landmark_val2ind[pts[2]]
        vert2ind.append((x,y,z))    

    tri_val2ind,tri_ind2val = create_dict(vert2ind)





frame2 = cv2.imread("img_vid/jim_carrey.jpg",-1)
gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
faces = detector(gray2)
for face in faces:

    landmarks2 = predictor(gray2, face)
    landmark_lst2 = []
    for n in range(0, 68):
        x = landmarks2.part(n).x
        y = landmarks2.part(n).y
        landmark_lst2.append((x,y))
        #cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)

    landmark_lst_32_2 = np.array(landmark_lst2,dtype=np.int32)
    landmark_val2ind2, landmark_ind2val2 = create_dict(landmark_lst2)
    convexHull = cv2.convexHull(landmark_lst_32_2)
    rect_2 = cv2.boundingRect(convexHull)   


last= list(tri_ind2val.items())[-1][0]+1

f1 = np.zeros_like(frame)
f2 = np.zeros_like(frame2)

for i in range(0,last):
    point1,point2,point1_aff,point2_aff = triangle(i)
    
    cropped1,w1,h1,x1,y1 = crop(frame,point1)  #ORIG
    cropped2,w2,h2,x2,y2 = crop(frame2,point2)

    # cropped1,w1,h1,x1,y1 = crop(frame,point1,point1_aff)   #EXP
    # cropped2,w2,h2,x2,y2 = crop(frame2,point2,point2_aff)

    point1_f = np.float32(point1_aff)
    point2_f = np.float32(point2_aff)


    M1 = cv2.getAffineTransform(point1_f,point2_f)
    M2 = cv2.getAffineTransform(point2_f,point1_f)

    warped_tri_1 = cv2.warpAffine(cropped1,M1,(w2,h2))
    warped_tri_2 = cv2.warpAffine(cropped2,M2,(w1,h1))

    #frame[y1:y1+h1,x1:x1+w1] = cv2.addWeighted(warped_tri_2,1,frame[y1:y1+h1,x1:x1+w1],1,0,)
    #frame2[y2:y2+h2,x2:x2+w2] = cv2.addWeighted(warped_tri_1,1,frame2[y2:y2+h2,x2:x2+w2],1,0,)

    f1 = overlay(f1,warped_tri_2,x1,y1,w1,h1)
    f2 = overlay(f2,warped_tri_1,x2,y2,w2,h2)

    #frame[y1:y1+h1,x1:x1+w1] = cv2.add(frame[y1:y1+h1,x1:x1+w1],warped_tri_2)
    #frame2[y2:y2+h2,x2:x2+w2] = cv2.add(frame2[y2:y2+h2,x2:x2+w2],warped_tri_1)

f1 = cv2.morphologyEx(f1,cv2.MORPH_OPEN,(4,4),iterations=3)
f2 = cv2.morphologyEx(f2,cv2.MORPH_OPEN,(4,4),iterations=3)


frame_overlayed,mask1 = final_overlay(frame,f1)
frame2_overlayed,mask2 = final_overlay(frame2,f2)

# cv2.imshow("result",frame_overlayed)
# cv2.imshow("img2",frame)
# cv2.imshow("img2_head_mask",mask1)
# cv2.waitKey(0)

frame_final = seamless(frame_overlayed,frame,mask1,rect_1)
frame2_final = seamless(frame2_overlayed,frame2,mask2,rect_2)

#cv2.imshow("f1",f1)
#cv2.imshow("f2",f2)    
cv2.imshow("Frame1", frame_final)
cv2.imshow("Frame2", frame2_final)


key = cv2.waitKey(0)


