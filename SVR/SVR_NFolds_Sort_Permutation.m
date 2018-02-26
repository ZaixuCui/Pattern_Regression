
function SVR_NFolds_Sort_Permutation(Subjects_Data_Path, Subjects_Scores, Times, FoldQuantity, C_Range, ResultantFolder)

for i = 1:Times
   
    ResultantFolder_I = [ResultantFolder filesep 'Time_' num2str(i)];
    mkdir(ResultantFolder_I);
    
    Job_Name = ['perm_' num2str(i)];
    pipeline.(Job_Name).command = 'SVR_NFolds_Sort_Permutation_Sub(opt.para1, opt.para2, '''', opt.para3, ''Scale'', opt.para4, opt.para5)';
    pipeline.(Job_Name).opt.para1 = Subjects_Data_Path;
    pipeline.(Job_Name).opt.para2 = Subjects_Scores;
    pipeline.(Job_Name).opt.para3 = FoldQuantity;
    pipeline.(Job_Name).opt.para4 = C_Range;
    pipeline.(Job_Name).opt.para5 = ResultantFolder_I;

end

psom_gb_vars;
Pipeline_opt.mode = 'batch';
Pipeline_opt.qsub_options = '-q veryshort.q';
Pipeline_opt.mode_pipeline_manager = 'batch';
Pipeline_opt.max_queued = 40;
Pipeline_opt.flag_verbose = 1;
Pipeline_opt.flag_pause = 0;
Pipeline_opt.flag_update = 1;
Pipeline_opt.path_logs = [ResultantFolder filesep 'logs'];

psom_run_pipeline(pipeline,Pipeline_opt);


