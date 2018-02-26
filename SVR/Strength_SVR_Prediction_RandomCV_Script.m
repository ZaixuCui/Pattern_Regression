
clear

ParentFolder = '/HeLabData/ztd/cuizaixu/DATA_HCP_AlgorithmsCompare_Revise/REST1_GR_Strength_RandomCV/';
REST_Data_Path =  [ParentFolder '/FC_REST1_GlobalRegress_NonNAN.mat'];
load ../../Strength_AgeAdj_NonNAN.mat;
load ../../SampleInfo_Strength_AgeAdj.mat;
ResultantFolder = [ParentFolder '/FC/SVR'];

for i = 25:length(SampleInfo_Strength_AgeAdj)
    ResultantFolder_I = [ResultantFolder filesep 'SampleSize_' num2str(SampleInfo_Strength_AgeAdj(i).Size)];
    mkdir(ResultantFolder_I);
    for j = 1:50
        Job_Name = ['SVR_' num2str(i) '_' num2str(j)];
        pipeline.(Job_Name).command = 'SVR_NFolds_RandomCV_ForSubset(opt.para1, opt.para2, 5, 2, ''Scale'', opt.para5, opt.para6, opt.para7, opt.para8)';
        pipeline.(Job_Name).opt.para1 = REST_Data_Path;
        pipeline.(Job_Name).opt.para2 = Strength_AgeAdj_NonNAN;
        pipeline.(Job_Name).opt.para5 = j;
        pipeline.(Job_Name).opt.para6 = SampleInfo_Strength_AgeAdj(i).Index{j};
        pipeline.(Job_Name).opt.para7 = power(2, [-5:10]);
        pipeline.(Job_Name).opt.para8 = ResultantFolder_I;
    end
end

Pipeline_opt.mode = 'qsub';
Pipeline_opt.qsub_options = '-q he_queue.q';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 200;
Pipeline_opt.flag_verbose = 1;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.flag_update = 1;
Pipeline_opt.path_logs = [ResultantFolder filesep 'logs2'];

psom_run_pipeline(pipeline,Pipeline_opt);
