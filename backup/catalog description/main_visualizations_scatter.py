from backup.src.visualization.scatter_generic import execute_visualization_scatter

execute_visualization_scatter(
    in_file_path='statistics/catalog_statistics_per_model_no_outliers.csv',
    out_dir_path='visualizations/per_model_no_outliers',
    selected_combinations=[("total_classes","total_relations"),
("stereotyped_classes","stereotyped_relations"),
("non_stereotyped_classes","non_stereotyped_relations"),
("ontouml_classes","ontouml_relations"),
("non_ontouml_classes","non_ontouml_relations")])
