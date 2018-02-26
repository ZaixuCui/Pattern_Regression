
clear

ParentFolder = '/mnt/data4/cuizaixu/DATA_HCP_AlgorithmsCompare_Revise/Cluster_Coefficient/REST1_GR_Strength';
REST_Data_Path =  [ParentFolder '/ClusterCoefficient_REST1_GlobalRegress_NonNAN.mat'];
load([ParentFolder '/Strength_AgeAdj_NonNAN.mat']);
load([ParentFolder '/SampleInfo_Strength_AgeAdj.mat']);
ResultantFolder = [ParentFolder '/RVR'];

for i = 1:length(SampleInfo_Strength_AgeAdj)
    ResultantFolder_I = [ResultantFolder filesep 'SampleSize_' num2str(SampleInfo_Strength_AgeAdj(i).Size)];
    mkdir(ResultantFolder_I);
    for j = 1:50
        Job_Name = ['RVR_' num2str(i) '_' num2str(j)];
        pipeline.(Job_Name).command = 'RVR_NFolds_Sort_ForSubset(opt.para1, opt.para2, 5, ''Scale'', opt.para5, opt.para6, opt.para7)';
        pipeline.(Job_Name).opt.para1 = REST_Data_Path;
        pipeline.(Job_Name).opt.para2 = Strength_AgeAdj_NonNAN';
        pipeline.(Job_Name).opt.para5 = j;
        pipeline.(Job_Name).opt.para6 = SampleInfo_Strength_AgeAdj(i).Index{j};
        pipeline.(Job_Name).opt.para7 = ResultantFolder_I;
    end
end

Pipeline_opt.mode = 'batch';
Pipeline_opt.qsub_options = '-q long -l nodes=1:ppn=1';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 50;
Pipeline_opt.flag_verbose = 1;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.flag_update = 1;
Pipeline_opt.path_logs = [ResultantFolder filesep 'logs'];

psom_run_pipeline(pipeline,Pipeline_opt);
