from __future__ import print_function # dùng để hỗ trợ cả python 2 và python 3
import numpy as np # thư viện dùng để tính toán các số liệu xung quanh dữ liệu dạng mảng
import matplotlib.pyplot as plt # dùng để biểu diễn vẽ biểu đồ hiện thị dl
from scipy.spatial.distance import cdist # tính khoảng cách giữa các điểm dữ liệu
np.random.seed(11) #tạo một dãy số ngẫu nhiên mỗi lần chạy, nhưng nếu có seed thì sẽ giữ nguyên dãy số ngẫu nhiên đó vào những lần tiếp theo chạy
means = [[2,2], [8,3], [3,6]] # tạo tâm cho các cụm dữ liệu 
cov = [[1,0], [0,1]] # tạo ma trận quy định các điểm phân tán ntn
N = 500 # số lượng điểm của từng cụm
X0 = np.random.multivariate_normal(means[0], cov, N) # tạo cụm dữ liệu theo phân phối chuẩn 
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)
X = np.concatenate((X0, X1, X2), axis = 0) # gộp các cụm lại thành một tập dữ liệu theo hàng
K = 3 # số lượng cụm
original_label = np.asarray([0]*N + [1]*N + [2]*N).T # tạo nhãn cho từng điểm
def kmeans_display(X, label):
    K = np.amax(label) + 1 #tính số cụm
    X0 = X[label == 0, :] # lấy ra các điểm thuộc 0
    X1 = X[label == 1, :] # lấy ra các điểm thuộc 1
    X2 = X[label == 2, :] # lấy ra các điểm thuộc 2
    plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8) # vẽ các điểm thuộc cụm 0
    plt.plot(X1[:, 0], X1[:, 1], 'go', markersize = 4, alpha = .8) # vẽ các điểm thuộc cụm 1
    plt.plot(X2[:, 0], X2[:, 1], 'rs', markersize = 4, alpha = .8) # vẽ các điểm thuộc cụm 2
    plt.axis('equal') # tạo tỉ lệ trục x và y bằng nhau
    plt.plot() 
    plt.show() # hiển thị biểu đồ
kmeans_display(X, original_label) # hiển thị dữ liệu gốc
def kmeans_init_centers(X, k): # hàm chọn tâm 
    return X[np.random.choice(X.shape[0], k, replace = False)]
def kmeans_assign_labels(X, centers): # hàm gán nhãn mới khi biết các center
    D = cdist(X, centers) # khoảng cách giữa các điểm dữ liệu và tâm cụm
    return np.argmin(D, axis = 1) # trả về vị trí nhỏ nhất trong mảng 
def kmeans_update_centers(X, labels, K): # hàm cập nhật lại tâm mới khi biết nhãn mới
    centers = np.zeros((K, X.shape[1])) # tạo mảng để lưu tâm mới
    for k in range(K): # duyệt qua từng cụm
        Xk = X[labels == k, :] # lấy ra các điểm thuộc cụm k
        centers [k, :] = np.mean(Xk, axis = 0) # tính trung bình các điểm thuộc cụm 
    return centers # trả về tâm mới
def has_converged(centers, new_centers): # hàm kiểm tra xem các tâm có hội tụ chưa
    return (set([tuple(a) for a in centers]) == set([tuple(a) for a in new_centers])) # so sánh các tâm cũ và tâm mới
def kmeans(X, K0):
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1])) # gán nhãn mới lấy phân tử cuối cùng trong danh sách làm tâm
        new_center = kmeans_update_centers(X, labels[it], K) # cập nhật tâm mới
        if has_converged(centers[-1], new_center): # kiểm tra xem các tâm có hội tụ chưa
            break
        centers.append(new_center) # thêm tâm mới vào danh sách tâm
        it += 1
    return (centers, labels, it)
(centers, labels, it) = kmeans(X, K)
print('Centers found by our algorithm:')
print(centers[-1])

kmeans_display(X, labels[-1])