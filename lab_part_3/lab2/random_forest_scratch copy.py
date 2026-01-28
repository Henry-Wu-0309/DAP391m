
import numpy as np
from collections import Counter

class Node:
    """
    Một Node trong cây quyết định (Decision Tree).
    Lưu trữ thông tin về đặc trưng chia (feature), ngưỡng chia (threshold),
    các con trái/phải, hoặc giá trị dự đoán nếu là lá (leaf).
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature      # Index của đặc trưng dùng để chia
        self.threshold = threshold  # Giá trị ngưỡng để chia
        self.left = left            # Nhánh con bên trái
        self.right = right          # Nhánh con bên phải
        self.value = value          # Giá trị dự đoán (chỉ dùng nếu node này là node lá)

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeRegressor:
    """
    Cây quyết định hồi quy (Regression Tree) tự cài đặt.
    """
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features # Số lượng đặc trưng ngẫu nhiên được chọn tại mỗi lần chia
        self.root = None

    def fit(self, X, y):
        # Nếu không quy định số đặc trưng thì dùng tất cả
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Điều kiện dừng:
        # 1. Đạt độ sâu tối đa
        # 2. Chỉ còn 1 giá trị duy nhất
        # 3. Số lượng mẫu ít hơn mức tối thiểu để chia
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Chọn ngẫu nhiên các đặc trưng (Feature Selection - đặc trưng của Random Forest)
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Tìm cách chia tốt nhất
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # Nếu không tìm được cách chia hợp lý, tạo node lá
        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # Thực hiện chia dữ liệu
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # Tính toán khả năng giảm phương sai (Variance Reduction)
                gain = self._variance_reduction(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = thr

        return split_idx, split_thresh

    def _variance_reduction(self, y, X_column, threshold):
        # Tính Variance ban đầu
        parent_var = np.var(y)

        # Chia thành con trái/phải
        left_idxs, right_idxs = self._split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Tính Variance có trọng số của các con
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = np.var(y[left_idxs]), np.var(y[right_idxs])
        child_var = (n_l / n) * e_l + (n_r / n) * e_r

        # Gain = Variance cha - Variance con
        return parent_var - child_var

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _calculate_leaf_value(self, y):
        # Giá trị dự đoán tại lá là trung bình cộng (cho bài toán hồi quy)
        return np.mean(y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestRegressorScratch:
    """
    Thuật toán Random Forest (Rừng ngẫu nhiên) tự cài đặt.
    """
    def __init__(self, n_estimators=10, min_samples_split=2, max_depth=100, n_features=None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_estimators):
            # Cài đặt cây quyết định mới
            tree = DecisionTreeRegressor(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features
            )
            
            # --- Bước 1: Bootstrapping (Lấy mẫu có lặp lại) ---
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            # --- Bước 2: Huấn luyện cây trên mẫu bootstrap ---
            # Lưu ý: Việc chọn ngẫu nhiên features đã được xử lý bên trong 
            # hàm _grow_tree của class DecisionTreeRegressor
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        # --- Bước 3: Aggregation (Tổng hợp kết quả) ---
        # Lấy dự đoán từ tất cả các cây
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Kết quả cuối cùng là trung bình cộng của tất cả các cây (Hồi quy)
        # Nếu là phân loại (Classification), ta sẽ dùng bầu chọn đa số (Majority Voting)
        tree_avg = np.mean(predictions, axis=0)
        return tree_avg

# --- Phần chạy thử nghiệm (Demo) ---
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error

    # 1. Đọc dữ liệu (giống notebook)
    try:
        df = pd.read_csv('data.csv')
        
        # Xử lý nhanh dữ liệu giống notebook
        df = pd.get_dummies(df)
        labels = np.array(df['actual'])
        features = df.drop('actual', axis=1)
        feature_list = list(features.columns)
        features = np.array(features)
        
        # Chia train/test
        train_features, test_features, train_labels, test_labels = train_test_split(
            features, labels, test_size=0.25, random_state=42
        )
        
        print("Dữ liệu đã sẵn sàng:")
        print(f"Train shapes: {train_features.shape}")
        print(f"Test shapes: {test_features.shape}")

        # 2. Khởi tạo và huấn luyện Random Forest Tự Code
        # n_estimators=10 để chạy cho nhanh, max_depth=5 để tránh overfitting
        # n_features='sqrt' (tự động tính căn bậc 2 số features)
        rf_scratch = RandomForestRegressorScratch(
            n_estimators=10, 
            max_depth=5,
            min_samples_split=2,
            n_features=int(np.sqrt(train_features.shape[1])) 
        )
        
        print("\nĐang huấn luyện Random Forest tự cài đặt...")
        rf_scratch.fit(train_features, train_labels)
        print("Huấn luyện xong!")

        # 3. Dự đoán và đánh giá
        predictions = rf_scratch.predict(test_features)
        mae = mean_absolute_error(test_labels, predictions)
        
        print(f"\nKết quả:")
        print(f"Mean Absolute Error (Tự code): {mae:.2f} degrees")
        
        # Để so sánh, accuracy kiểu notebook:
        errors = abs(predictions - test_labels)
        mape = 100 * (errors / test_labels)
        accuracy = 100 - np.mean(mape)
        print(f"Accuracy: {accuracy:.2f} %")
        
    except Exception as e:
        print(f"Có lỗi xảy ra khi chạy demo: {e}")
        print("Đảm bảo file 'data.csv' nằm cùng thư mục.")
