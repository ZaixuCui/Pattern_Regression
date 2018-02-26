
function [w_Brain, model_All] = W_Calculate_SVR_CSelect(Subjects_Data, Subjects_Scores, Covariates, Pre_Method, C_Range, ResultantFolder)
%
% Subject_Data:
%           m*n matrix
%           m is the number of subjects
%           n is the number of features
%
% Subject_Scores:
%           the continuous variable to be predicted,[1*m]
%
% Covariates:
%           m*n matrix
%           m is the number of subjects
%           n is the number of covariates
%
% Pre_Method:
%          'Normalize', 'Scale', 'None'
%
% ResultantFolder:
%           the path of folder storing resultant files
%

if nargin >= 3
    if ~exist(ResultantFolder, 'dir')
        mkdir(ResultantFolder);
    end
end

[Subjects_Quantity, Features_Quantity] = size(Subjects_Data);

if ~isempty(Covariates)
    [~, Covariates_quantity] = size(Covariates);
    M = 1;
    for j = 1:Covariates_quantity
        M = M + term(Covariates(:, j));
    end
    slm = SurfStatLinMod(Subjects_Data, M);
    
    Subjects_Data = Subjects_Data - repmat(slm.coef(1, :), Subjects_Quantity, 1);
    for j = 1:Covariates_quantity
        Subjects_Data = Subjects_Data - ...
            repmat(Covariates(:, j), 1, Features_Quantity) .* repmat(slm.coef(j + 1, :), Subjects_Quantity, 1);
    end
end

% Select optimal C
for m = 1:length(C_Range)
    Prediction_Inner = SVR_NFolds_Sort(Subjects_Data, Subjects_Scores, [], 5, Pre_Method, C_Range(m));
    Inner_Corr_Array(m) = Prediction_Inner.Mean_Corr;
    Inner_MAE_Array(m) = Prediction_Inner.Mean_MAE;
end
Inner_MAE_inv_Array = 1./Inner_MAE_Array;
Inner_Corr_norm_Array = (Inner_Corr_Array - mean(Inner_Corr_Array)) / std(Inner_Corr_Array);
Inner_MAE_inv_norm_Array = (Inner_MAE_inv_Array - mean(Inner_MAE_inv_Array)) / std(Inner_MAE_inv_Array);
Inner_Evaluation = Inner_Corr_norm_Array + Inner_MAE_inv_norm_Array;
[~, Max_Index] = max(Inner_Evaluation);
C_Optimal = C_Range(Max_Index);

if strcmp(Pre_Method, 'Normalize')
    %Normalizing
    MeanValue = mean(Subjects_Data);
    StandardDeviation = std(Subjects_Data);
    [~, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MeanValue(j)) / StandardDeviation(j);
    end
elseif strcmp(Pre_Method, 'Scale')
    % Scaling to [0 1]
    MinValue = min(Subjects_Data);
    MaxValue = max(Subjects_Data);
    [~, columns_quantity] = size(Subjects_Data);
    for j = 1:columns_quantity
        Subjects_Data(:, j) = (Subjects_Data(:, j) - MinValue(j)) / (MaxValue(j) - MinValue(j));
    end
end
    
% SVR
Subjects_Scores = Subjects_Scores';
Subjects_Data = double(Subjects_Data);
model_All = svmtrain(Subjects_Scores, Subjects_Data, ['-s 3 -t 0 -c ' num2str(C_Optimal)]);
w_Brain = zeros(1, Features_Quantity);
for j = 1 : model_All.totalSV
    w_Brain = w_Brain + model_All.sv_coef(j) * model_All.SVs(j, :);
end

w_Brain = w_Brain / norm(w_Brain);

if nargin >= 6
    save([ResultantFolder filesep 'w_Brain.mat'], 'w_Brain');
end
