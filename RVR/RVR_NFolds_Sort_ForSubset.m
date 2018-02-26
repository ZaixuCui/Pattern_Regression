
function RVR_NFolds_Sort_ForSubset(Subjects_Data_Path, Subjects_Scores, FoldQuantity, Pre_Method, SampleIndex, SelectedIDs, ResultantFolder)

tmp = load(Subjects_Data_Path);
FieldName = fieldnames(tmp);
Subjects_Data = tmp.(FieldName{1});

Scores_Selected = Subjects_Scores(SelectedIDs);
Data_Selected = Subjects_Data(SelectedIDs, :);

Prediction = RVR_NFolds_Sort(Data_Selected, Scores_Selected, [], FoldQuantity, Pre_Method, 0);
save([ResultantFolder '/PredictionAllInfo_' num2str(SampleIndex) '.mat'], 'Prediction');
Mean_Corr = Prediction.Mean_Corr;
Mean_MAE = Prediction.Mean_MAE;
save([ResultantFolder '/Prediction_' num2str(SampleIndex) '.mat'], 'Mean_Corr', 'Mean_MAE', 'SelectedIDs');
