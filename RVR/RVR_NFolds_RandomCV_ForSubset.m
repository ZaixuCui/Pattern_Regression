
function RVR_NFolds_RandomCV_ForSubset(Subjects_Data_Path, Subjects_Scores, FoldQuantity, CVRepeatTimes, Pre_Method, SampleIndex, SelectedIDs, ResultantFolder)

tmp = load(Subjects_Data_Path);
FieldName = fieldnames(tmp);
Subjects_Data = tmp.(FieldName{1});

Scores_Selected = Subjects_Scores(SelectedIDs);
Data_Selected = Subjects_Data(SelectedIDs, :);

Prediction = RVR_NFolds_RandomCV(Data_Selected, Scores_Selected, FoldQuantity, CVRepeatTimes, Pre_Method);
save([ResultantFolder '/PredictionAllInfo_' num2str(SampleIndex) '.mat'], 'Prediction');
Mean_Corr = Prediction.CVMean_Corr;
Mean_MAE = Prediction.CVMean_MAE;
save([ResultantFolder '/Prediction_' num2str(SampleIndex) '.mat'], 'Mean_Corr', 'Mean_MAE', 'SelectedIDs');