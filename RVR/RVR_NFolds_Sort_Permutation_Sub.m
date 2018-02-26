
function Prediction = RVR_NFolds_Sort_Permutation_Sub(Subjects_Data_Path, Subjects_Scores, Covariates, FoldQuantity, Pre_Method, ResultantFolder)

tmp = load(Subjects_Data_Path);
FieldName = fieldnames(tmp);
Subjects_Data = tmp.(FieldName{1});
RVR_NFolds_Sort(Subjects_Data, Subjects_Scores, Covariates, FoldQuantity, Pre_Method, 0, 1, ResultantFolder)
