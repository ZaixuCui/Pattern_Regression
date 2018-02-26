
function SVR_NFolds_Sort_ForSubset(Subjects_Data_Path, Subjects_Scores, FoldQuantity, Pre_Method, SampleIndex, SelectedIDs, C_Range, ResultantFolder)

tmp = load(Subjects_Data_Path);
FieldName = fieldnames(tmp);
Subjects_Data = tmp.(FieldName{1});

Data_Selected = Subjects_Data(SelectedIDs, :);
Scores_Selected = Subjects_Scores(SelectedIDs);

Prediction = SVR_NFolds_Sort_CSelect(Data_Selected, Scores_Selected', [], FoldQuantity, Pre_Method, C_Range);
save([ResultantFolder '/PredictionAllInfo_' num2str(SampleIndex) '.mat'], 'Prediction');
Mean_Corr = Prediction.Mean_Corr;
Mean_MAE = Prediction.Mean_MAE;
save([ResultantFolder '/Prediction_' num2str(SampleIndex) '.mat'], 'Mean_Corr', 'Mean_MAE', 'SelectedIDs');
