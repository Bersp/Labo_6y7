import cv2
import numpy as np

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

def calibrate_camera(dset, corner_size, calsize):
    """
    Calculates the camera parameters, including distortion coefficients and the
    undistort maps.
    """

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # number of inner corners is Cx x Cy, this should be an input from user!
    Cx, Cy = corner_size 

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(Cy,5,0)
    objp = np.zeros((Cx*Cy,3), np.float32)
    objp[:,:2] = np.mgrid[0:Cx,0:Cy].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    Lx, Ly, Nimages = dset.shape 

    for i in range(Nimages):
        img = dset[:, :, i]
        # turn into 8 bit gray 
        img = (img/4).astype('uint8')
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(img, (Cx, Cy), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (Cx, Cy), corners2,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, \
            img.shape[::-1], None, None)
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(Ly,Lx),1,(Ly,Lx))
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (Ly, Lx), 5)
    return newcameramtx, roi, mapx, mapy


def undistort_image(img, mapx, mapy):
    """ 
    Undistorts image using calibration data.
    """
    
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# # Reprojection error estimation
# tot_error = 0
# for i in range(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
#     tot_error += error
# print( "mean error: ", tot_error/len(objpoints))


def calibrate_accelerometer(dataX, dataY, dataZ, gval):
    """
    Calibrates accelerometer, taking a number of postures as input.
    """

    # % Tengo 9 mediciones en total. Cada una tiene las aceleraciones en los 3
    # % ejes. Tomo la aceleracion de cada medicion en cada eje y promedio ese
    # % vector. 
    # % dataX,Y,Z tiene los promedios de las aceleraciones en X,Y,Z para todas
    # % las mediciones

    # %dataX = [mean(data1(:,1)), mean(data2(:,1)), mean(data3(:,1)), mean(data4(:,1)), mean(data5(:,1)), mean(data6(:,1)),mean(data7(:,1)), mean(data8(:,1)),mean(data9(:,1))];
    # %dataY = [mean(data1(:,2)), mean(data2(:,2)), mean(data3(:,2)), mean(data4(:,2)), mean(data5(:,2)), mean(data6(:,2)),mean(data7(:,2)), mean(data8(:,2)),mean(data9(:,2))];
    # %dataZ = [mean(data1(:,3)), mean(data2(:,3)), mean(data3(:,3)), mean(data4(:,3)), mean(data5(:,3)), mean(data6(:,3)),mean(data7(:,3)), mean(data8(:,3)),mean(data9(:,3))];

    # N = length(dataX);
    # zeta = zeros(N,6);
    # zeta(:,1) = dataX(:).^2;
    # zeta(:,2) = dataY(:).^2;
    # zeta(:,3) = dataZ(:).^2;
    # zeta(:,4) = -2*dataX(:);
    # zeta(:,5) = -2*dataY(:);
    # zeta(:,6) = -2*dataZ(:);

    # g = (gval^2)*ones(N,1);

    # xi = pinv(transpose(zeta)*zeta)*transpose(zeta)*g;
    zetaT = np.transpose(zeta)
    xi = np.linalg.pinv( zetaT*zeta )*zetaT*g

    # C = 1+ xi(4)^2/xi(1) + xi(5)^2/xi(2) + xi(6)^2/xi(3);
    C = 1 + xi[3]**2/xi[0] + xi[4]**2/xi[1] + xi[5]**2/xi[2]

    # S = sqrt(C./xi(1:3));
    S = np.sqrt(C/xi[0:2])
    
    # O = xi(4:6)./xi(1:3);
    O = xi[3:5]/xi[0:2]

    # S = S.' ;
    # O = O.' ;

    S = np.abs(S);

    # %disp('OFFSET: ' num2str(O(1)) ' ' num2str(O(2)) ' ' num2str(O(3)) ])
    # %disp('SENSIT: ' num2str(S(1)) ' ' num2str(S(2)) ' ' num2str(S(3)) ])


