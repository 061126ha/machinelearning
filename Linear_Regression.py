# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals
import numpy as np 
import matplotlib.pyplot as plt 

# height (cm)
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T # .T chuyển vị trí 
# weight (kg)
y = np.array([[ 49, 50, 51,  54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T
# Visualize data 
plt.plot(X, y, 'ro') # dùng để vẽ các điểm trên biểu đồ r màu đỏ o hình tròn 
plt.axis([140, 190, 45, 75]) # giới hạn biểu đồ X từ 140 - 170, Y từ 45 - 75 
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show() # hiện thị trên bản đồ

# Building Xbar
one = np.ones((X.shape[0], 1)) # tạo ma trận 1 có số hàng bằng số hàng của X, số cột = 1
Xbar = np.concatenate((one, X), axis = 1) # ghép ma trận one và X theo chiều ngang (axis=1)
# Calculating weights of the fitting line
A = np.dot(Xbar.T, Xbar) # tính tích vô hướng của ma trận Xbar chuyển vị và ma trận Xbar
b = np.dot(Xbar.T, y) # tính tích vô hướng của ma trận Xbar chuyển vị và ma trận y
w = np.dot(np.linalg.pinv(A), b) # tính trọng số w bằng cách nhân ma trận nghịch đảo của A với b
# Predicting the fiting line
w_0 = w[0][0] # trọng số w_0 là phần tử đầu tiên của ma trận w
w_1 = w[1][0] # trọng số w_1 là phần tử thứ hai của ma trận w
x0 = np.linspace(140, 190, 2) # tạo một mảng x0 từ 140 đến 190 với 2 phần tử
y0 = w_0 + w_1*x0 # tính giá trị y0 bằng công thức y = w_0 + w_1*x0

# Drawing the fitting line
plt.plot(X.T, y.T, 'ro') # vẽ các điểm dữ liệu gốc
plt.plot(x0, y0) # vẽ đường hồi quy tuyến tính
plt.axis([140, 190, 45, 75]) # giới hạn biểu đồ X từ 140 - 190, Y từ 45 - 75
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.show() # hiện thị trên bản đồ

y1 = w_0 + w_1*155 # dự đoán giá trị y1 khi x=155
y2 = w_0 + w_1*160 # dự đoán giá trị y2 khi x=160
print( u'Predicted weight of a person with height 155 cm: %.2f (kg), real number: 52 (kg)' %(y1) )
print( u'Predicted weight of a person with height 160 cm: %.2f (kg), real number: 56 (kg)' %(y2) )
