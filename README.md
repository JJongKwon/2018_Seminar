# Seminar
2018 seminar
<table>
    <thead>
        <tr>
            <th>model : [Generalized Linear Models](http://scikit-learn.org/stable/modules/linear_model.html)</th>
            <th>module</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[Ordinary Least Squares](http://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)</td>
            <td>[linear_model.LinearRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)</td>
            <td>Ordinary least squares Linear Regression.</td>
        </tr>
        <tr>
            <td rowspan=3>[Ridge](http://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)</td>
            <td>[linear_model.Ridge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge)</td>
            <td>Linear least squares with l2 regularization.</td>
        </tr>
        <tr>
            <td>[linear_model.RidgeCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#sklearn.linear_model.RidgeCV)</td>
            <td>Ridge regression with built-in cross-validation.</td>
        </tr>
        <tr>
            <td>[linear_model.ridge_regression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html#sklearn.linear_model.ridge_regression)</td>
            <td>Solve the ridge equation by the method of normal equations.</td>
        </tr>
        <tr>
            <td rowspan=2>[Lasso](http://scikit-learn.org/stable/modules/linear_model.html#lasso)</td>
            <td>[linear_model.Lasso](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html#sklearn.linear_model.Lasso)</td>
            <td>Linear Model trained with L1 prior as regularizer (aka the Lasso)</td>
        </tr>
        <tr>
            <td>[linear_model.LassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV)</td>
            <td>Lasso linear model with iterative fitting along a regularization path</td>
        </tr>
        <tr>
            <td>[Multi-task Lasso](http://scikit-learn.org/stable/modules/linear_model.html#multi-task-lasso)</td>
            <td>[linear_model.MultiTaskLassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html#sklearn.linear_model.MultiTaskLassoCV)</td>
            <td>Multi-task L1/L2 Lasso with built-in cross-validation.</td>
        </tr>
        <tr>
            <td rowspan=2>[Elastic Net](http://scikit-learn.org/stable/modules/linear_model.html#elastic-net)</td>
            <td>[linear_model.ElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet)</td>
            <td>Linear regression with combined L1 and L2 priors as regularizer.</td>
        </tr>        
        <tr>
            <td>[linear_model.ElasticNetCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html#sklearn.linear_model.ElasticNetCV)</td>
            <td>Elastic Net model with iterative fitting along a regularization path</td>
        </tr>        
        <tr>
            <td>[Multi-task Elastic Net](http://scikit-learn.org/stable/modules/linear_model.html#multi-task-elastic-net)</td>
            <td>[linear_model.MultiTaskElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNet.html#sklearn.linear_model.MultiTaskElasticNet)</td>
            <td>Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer</td>
        </tr>        
        <tr>
            <td>[Least Angle Regression](http://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression)</td>
            <td>[linear_model.Lars](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html#sklearn.linear_model.Lars)</td>
            <td>Least Angle Regression model a.k.a.</td>
        </tr>        
        <tr>
            <td rowspan=3>[LARS Lasso](http://scikit-learn.org/stable/modules/linear_model.html#lars-lasso)</td>
            <td>[linear_model.LassoLars](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars)</td>
            <td>Lasso model fit with Least Angle Regression a.k.a.</td>
        </tr>        
        <tr>
            <td>[linear_model.LassoLarsCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#sklearn.linear_model.LassoLarsCV)</td>
            <td>Cross-validated Lasso, using the LARS algorithm</td>
        </tr>        
        <tr>
            <td>[linear_model.LassoLarsIC](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#sklearn.linear_model.LassoLarsIC)</td>
            <td>Lasso model fit with Lars using BIC or AIC for model selection</td>
        </tr>        
        <tr>
            <td rowspan=4>[Orthogonal Matching Pursuit (OMP)](http://scikit-learn.org/stable/modules/linear_model.html#orthogonal-matching-pursuit-omp)</td>
            <td>[linear_model.OrthogonalMatchingPursuit](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit)</td>
            <td>Orthogonal Matching Pursuit model (OMP)</td>
        </tr>        
        <tr>
            <td>[linear_model.OrthogonalMatchingPursuitCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#sklearn.linear_model.OrthogonalMatchingPursuitCV)</td>
            <td>Cross-validated Orthogonal Matching Pursuit model (OMP)</td>
        </tr>        
        <tr>
            <td>[linear_model.orthogonal_mp](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp.html#sklearn.linear_model.orthogonal_mp)</td>
            <td>Orthogonal Matching Pursuit (OMP)</td>
        </tr>        
        <tr>
            <td>[linear_model.orthogonal_mp_gram](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.orthogonal_mp_gram.html#sklearn.linear_model.orthogonal_mp_gram)</td>
            <td>Gram Orthogonal Matching Pursuit (OMP)
</td>
        </tr>
        <tr>
            <td rowspan=2>[Bayesian Regression](http://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression)</td>
            <td>[linear_model.BayesianRidge](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge)</td>
            <td>[Bayesian ridge regression](http://scikit-learn.org/stable/modules/linear_model.html#bayesian-ridge-regression)</td>
        </tr>
        <tr>
            <td>[linear_model.ARDRegression](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression)</td>
            <td>[Bayesian ARD regression](http://scikit-learn.org/stable/modules/linear_model.html#automatic-relevance-determination-ard)</td>
        </tr>
        <tr>
            <td>[Stochastic Gradient Descent(SGD)](http://scikit-learn.org/stable/modules/sgd.html#stochastic-gradient-descent)</td>
            <td>[linear_model.SGDRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor)</td>
            <td>Linear model fitted by minimizing a regularized empirical loss with SGD</td>
        </tr>
        <tr>
            <td>[Passive Aggressive Algorithms](http://scikit-learn.org/stable/modules/linear_model.html#passive-aggressive-algorithms)</td>
            <td>[linear_model.PassiveAggressiveRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor)</td>
            <td>Passive Aggressive Regressor</td>
        </tr>
        <tr>
            <td rowspan=3>[Robustness regression](http://scikit-learn.org/stable/modules/linear_model.html#robustness-regression-outliers-and-modeling-errors)</td>
            <td>[linear_model.RANSACRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RANSACRegressor.html#sklearn.linear_model.RANSACRegressor)</td>
            <td>[RANSAC (RANdom SAmple Consensus) algorithm](http://scikit-learn.org/stable/modules/linear_model.html#ransac-random-sample-consensus)</td>
        </tr>
        <tr>
            <td>[linear_model.TheilSenRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor)</td>
            <td>[Theil-Sen Estimator: robust multivariate regression model](http://scikit-learn.org/stable/modules/linear_model.html#theil-sen-estimator-generalized-median-based-estimator)</td>
        </tr>
        <tr>
            <td>[linear_model.HuberRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor)</td>
            <td>[Linear regression model that is robust to outliers](http://scikit-learn.org/stable/modules/linear_model.html#huber-regression)</td>
        </tr>
        <tr>
            <td colspan=3>[Polynomial regression: extending linear models with basis functions](http://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)</td>
        </tr>
        <tr>
            <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
        </tr>
    </tbody>
     <thead>
        <tr>
            <th>model : [Kernel ridge regression](http://scikit-learn.org/stable/modules/kernel_ridge.html#kernel-ridge-regression)</th>
            <th>module : [kernel_ridge.KernelRidge](http://scikit-learn.org/stable/modules/generated/sklearn.kernel_ridge.KernelRidge.html#sklearn.kernel_ridge.KernelRidge)</th>
            <th> </th>
        </tr>
        <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    </thead>
    <tbody>
    <thead>
        <tr>
            <th>model : [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html#support-vector-machines)</th>
            <th>module</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>[Support Vector Regression](http://scikit-learn.org/stable/modules/svm.html#regression)</td>
            <td>[svm.LinearSVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR)</td>
            <td>Linear Support Vector Regression.</td>
        </tr>
        <tr>
            <td>[svm.NuSVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR)</td>
            <td>Nu Support Vector Regression.</td>
        </tr>
        <tr>
            <td>[svm.SVR](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)</td>
            <td>Epsilon-Support Vector Regression.</td>
        </tr>
    </tbody>
    <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
        <thead>
        <tr>
            <th>model : [Nearest Neighbors](http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors)</th>
            <th>module</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>[Nearest Neighbors Regression](http://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression)</td>
            <td>[neighbors.KNeighborsRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html#sklearn.neighbors.KNeighborsRegressor)</td>
            <td>Regression based on k-nearest neighbors.</td>
        </tr>
        <tr>
            <td>[neighbors.RadiusNeighborsRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor)</td>
            <td>Regression based on neighbors within a fixed radius.</td>
        </tr>
    </tbody>
    <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    <thead>
        <tr>
            <th>model : [Gaussian Processes](http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-processes)</th>
            <th>module</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[Gaussian Process Regression (GPR)](http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr)</td>
            <td>[gaussian_process.GaussianProcessRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html#sklearn.gaussian_process.GaussianProcessRegressor)</td>
            <td>Gaussian process regression (GPR).</td>
        </tr>
    </tbody>
        <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    <thead>
        <tr>
            <th>model : [Cross decomposition](http://scikit-learn.org/stable/modules/cross_decomposition.html#cross-decomposition)</th>
            <th>module : [sklearn.cross_decomposition.PLSRegression](http://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html#sklearn-cross-decomposition-plsregression)</th>
            <th> </th>
        </tr>
    </thead>
                    <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    <thead>
        <tr>
            <th>model : [Decision Trees](http://scikit-learn.org/stable/modules/tree.html#decision-trees)</th>
            <th>module</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>[Regression](http://scikit-learn.org/stable/modules/tree.html#regression)</td>
            <td>[tree.DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor)</td>
            <td>A decision tree regressor.</td>
        </tr>
         <tr>
            <td>[tree.ExtraTreeRegressor](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree)</td>
            <td>An extremely randomized tree regressor.</td>
        </tr>
    </tbody>
                        <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    <thead>
        <tr>
            <th>model : [Ensemble Methods](http://scikit-learn.org/stable/modules/ensemble.html#ensemble-methods)</th>
            <th>module : [sklearn.ensemble](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble)</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>[Bagging](http://scikit-learn.org/stable/modules/ensemble.html#bagging-meta-estimator)</td>
            <td>[ensemble.BaggingRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html#sklearn.ensemble.BaggingRegressor)</td>
            <td>A Bagging regressor..</td>
        </tr>
         <tr>
            <td>[Random Forests](http://scikit-learn.org/stable/modules/ensemble.html#random-forests)</td>
            <td>[ensemble.RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)</td>
            <td>A random forest regressor.</td>
        </tr>
         <tr>
            <td>[AdaBoost](http://scikit-learn.org/stable/modules/ensemble.html#adaboost)</td>
            <td>[ensemble.AdaBoostRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor)</td>
            <td>An AdaBoost regressor.</td>
        </tr>
         <tr>
            <td>[Gradient Tree Boosting](http://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)</td>
            <td>[ensemble.GradientBoostingRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)</td>
            <td>Gradient Boosting for regression.</td>
        </tr>
    </tbody>
                        <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    <thead>
        <tr>
            <th>model : [Multiclass and multilabel algorithms](http://scikit-learn.org/stable/modules/multiclass.html#multiclass-and-multilabel-algorithms)</th>
            <th>module</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>[Multioutput regression](http://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression)</td>
            <td>[multioutput.MultiOutputRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor)</td>
            <td>Multi target regression.</td>
        </tr>
         <tr>
            <td>[multioutput.RegressorChain](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain)</td>
            <td>(base_estimator)A multi-label model that arranges regressions into a chain.</td>
        </tr>
    </tbody>
                            <td colspan=3> ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------</td>
    <thead>
        <tr>
            <th>model : [Isotonic regression](http://scikit-learn.org/stable/modules/isotonic.html#isotonic-regression)</th>
            <th>module : [isotonic.IsotonicRegression](http://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html#sklearn.isotonic.IsotonicRegression)</th>
            <th>explain</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=2>[Multioutput regression](http://scikit-learn.org/stable/modules/multiclass.html#multioutput-regression)</td>
            <td>[multioutput.MultiOutputRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputRegressor.html#sklearn.multioutput.MultiOutputRegressor)</td>
            <td>Multi target regression.</td>
        </tr>
         <tr>
            <td>[multioutput.RegressorChain](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html#sklearn.multioutput.RegressorChain)</td>
            <td>(base_estimator)A multi-label model that arranges regressions into a chain.</td>
        </tr>
    </tbody>
</table>
