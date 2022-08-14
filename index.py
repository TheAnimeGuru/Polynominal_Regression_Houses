import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

class Model:
    def __init__(self, alpha, lambda_, X_train, y_train):
        self.alpha = alpha
        self.lambda_ = lambda_
        self.X_train = X_train
        self.y_train = y_train
        self.w = np.zeros((X_train.shape[1])) # Initial weights
        self.b = 0.0 # Initial bias

        ## normalize X
        self.X_train, self.mu, self.sigma = self.z_score_normalize()
    
    # Squared Cost Function
    def Jwb(self):
        """
        Calculates squared loss of Function Fwb(x) => J(w,b)

        Args (self):
            X_train (ndarray (m,n,)): Data x, m datasets, n features
            y_train (ndarray (m,)): Real y, m datasets
            w (ndarray(n,)): Weight for features n
            b (scalar): Model bias
            lambda_ (scalar): Regularization Factor
        
        Returns:
            cost (scalar) : Squared Cost + Regularization Cost
        """
        m = self.X_train.shape[0]
        n = self.X_train.shape[1]
        cost = 0.0 
        reg_cost = 0.0

        # Usual Squared Cost
        for i in range(m):
            fwb_i = np.dot(self.X_train[i], self.w) + self.b
            cost += (fwb_i - self.y_train[i])**2
        cost /= (2*m)
        
        # Added Cost for regularization
        for j in range(n):
            reg_cost += self.w[j]**2
        reg_cost *= self.lambda_ / (2*m)
        cost += reg_cost

        return cost

    # Derived Cost Function for w
    def dw_jwb(self):
        """
        Calculates derived loss of Function Fwb(x) => J(w,b) for w

        Args (self):
            X_train (ndarray (m,n,)): Data x, m datasets, n features
            y_train (ndarray (m,)): Real y, m datasets
            w (ndarray(n,)): Weight for features n
            b (scalar): Model bias
            lambda_ (scalar): Regularization Factor
        
        Returns:
            cost (ndarray (n,)) : Derived Cost + Regularization Cost (for w)
        """
        m = self.X_train.shape[0]
        n = self.X_train.shape[1]
        cost = np.zeros((n)) 

        # Usual Cost
        for i in range(m):
            f_wb = np.dot(self.w, self.X_train[i]) + self.b
            cost += (f_wb - self.y_train[i]) * self.X_train[i]

        cost /= m

        # Regularization cost
        cost += (self.lambda_ / m) * self.w

        return cost
    
    # Derived Cost Function for b
    def db_jwb(self):
        """
        Calculates derived loss of Function Fwb(x) => J(w,b) for b

        Args (self):
            X_train (ndarray (m,n,)): Data x, m datasets, n features
            y_train (ndarray (m,)): Real y, m datasets
            w (ndarray(n,)): Weight for features n
            b (scalar): Model bias
        
        Returns:
            cost (scalar) : Derived Cost (for b)
        """
        m = self.X_train.shape[0]
        cost = 0.0

        # Usual Cost
        for i in range(m):
            f_wb = np.dot(self.w, self.X_train[i]) + self.b
            cost += f_wb - self.y_train[i]

        cost /= m

        return cost

    # Z-Score Normalization
    def z_score_normalize(self):
        """
        Normalizes features of X to be around [-1, 1] and symmeteric to each other

        Args (self):
            X(ndarray(m,n,)): Data X, m datasets, n features
        
        Returns:
            X_normalized(ndarray(m,n,)): X with z-score normalization
            mu(ndarray(n,)): Mean of n features from X
            sigma(ndarray(n,)): Standard deviation of n features from X
        """
        mu = np.mean(self.X_train, axis=0)
        sigma = np.std(self.X_train, axis=0)

        X_normalized = (self.X_train - mu) / sigma

        return X_normalized, mu, sigma
    
    # Prediction
    def predict(self, x):
        """
        Predicts y for model with given values for features x

        Args(self):
            w (ndarray(n,)): Model weight
            b (scalar): Model bias
        Args:
             x (ndarray (n,))  : Data x, n features
        
        Returns:
            y_hat (scalar): Prediction of y
        """
        return np.dot(self.w, x) + self.b

    # gradient descent
    def gradient_descent(self):
        """
        Executes Gradient Descent and returns updated values model parameters
        
        Args (self):
            w (ndarray(n,)): Initial model weight
            b (scalar): initial model bias
            alpha (float): Learning rate
        
        Returns:
            w (ndarray(n,)): Updated weight
            b (scalar): Updated bias
        """
        w_new = self.w - self.alpha * self.dw_jwb()
        b_new = self.b - self.alpha * self.db_jwb()

        return w_new, b_new

    # Train
    def train(self, iters, plot=True):
        """
        Trains the model for number of iters.
        
        Args (self):
            w (ndarray(n,)): Initial model weight
            b (scalar): initial model bias
        Args:
            iters (scalar): Amount of training steps
            plot (bool): If set to True, the result will be plotted.

        """
        weight_amnt = self.X_train.shape[1]
        cost_data = []
        for iter in range(iters):
            self.w, self.b = self.gradient_descent()
            cost = self.Jwb()
            cost_data.append(cost)

            if iter % 10 == 0:
                weight_info = ''
                for weight in range(self.w.shape[0]):
                    weight_info += f' Weight {weight}: {self.w[weight]:.4f},'
                print(f'Iteration: {iter}, Weights:{weight_info} Bias: {self.b:.4f} , Cost: {cost:.4f}')
        
        # Plotting data
        if plot:
            fig_grid_size = math.ceil(math.sqrt((weight_amnt+1)))
            fig, ax = plt.subplots(fig_grid_size, fig_grid_size)
            ax[0, 0].set_xlabel('Iters')
            ax[0, 0].set_ylabel('Cost')
            ax[0, 0].plot(cost_data)
            for i in range(weight_amnt):
                n = i+1
                fig_col = (n % fig_grid_size)
                fig_row = math.floor(n / fig_grid_size)
                ax[fig_row, fig_col].set_xlabel(df.columns[i])
                ax[fig_row, fig_col].set_ylabel("Price")
                ax[fig_row, fig_col].scatter(X[:, i], y, marker="x", color="red", label="target")
                ax[fig_row, fig_col].scatter(X[:, i], [self.predict(x) for x in self.X_train], label="predict")
                ax[fig_row, fig_col].legend()
            if weight_amnt < fig_grid_size**2:
                for n in range(weight_amnt, fig_grid_size**2):
                    fig_col = (n % fig_grid_size)
                    fig_row = math.floor(n / fig_grid_size)
                    ax[fig_row, fig_col].set_visible(False)
            fig.tight_layout()
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')
            plt.show()

if __name__ == '__main__':
    ### Hyperparams

    alpha = 0.1
    lambda_ = 1
    iters = 100

    ### Get data
    df = pd.read_csv("./data/data.csv")
    
    ## Prepare Data

    df.drop(
    labels=["date", "street", "city", "statezip", "country", "yr_built", "yr_renovated", "view", "waterfront"],
    axis=1,
    inplace=True
    )

    price_df = df['price']
    df.drop('price', axis=1, inplace=True)

    ## Create some polynominal Features
    for col in df.columns:
        df[f'squared_{col}'] = df[col]**2

    X = df.to_numpy(dtype=np.float64)
    y = price_df.to_numpy(dtype=np.float64)

    ## Train Model and Plot Results
    model = Model(alpha, lambda_, X, y)
    model.train(iters)
    
    for i in range(20):
        x = X[i*20]
        print(f'Price Difference for index {i*20}: {model.predict(x) - y[i*20]}$')
