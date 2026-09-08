from pathlib import Path
# #############################################
# constants
# #############################################
DATA_BASE_DIR = Path("/workdir1.8t/lw23/DataSet/EUS_MSITFF/")

# Created csv
CUSTOM_TRAIN_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/split1/selected_3class_eus_wle_combine_no1hospital_train.csv"
CUSTOM_VALID_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/split1/selected_3class_eus_wle_combine_no1hospital_valid.csv"
# CUSTOM_TEST_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/split1/selected_3class_eus_wle_combine_no1hospital_test - all.csv"
# CUSTOM_TEST_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/selected_3class_eus_wle_combine_1hospital_test - all.csv"
# CUSTOM_TEST_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/split1/selected_3class_eus_wle_combine_no1hospital_test - noReport.csv"
# CUSTOM_TEST_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/selected_3class_eus_wle_combine_1hospital_test - noReport.csv"

CUSTOM_TEST_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/split3/selected_3class_eus_wle_combine_no1hospital_test - noReport - noWLE.csv"
# CUSTOM_TEST_CSV = "/workdir1.8t/lw23/DataSet/EUS_MSITFF/6_3modify/selected_3class_eus_wle_combine_1hospital_test - noReport - noWLE.csv"

CUSTOM_PATH_COL_A = "Path_eus"
CUSTOM_PATH_COL_B = "Path_wle"

CUSTOM_REPORT_COL = "Report_Impression"

CUSTOM_LABELS_COL = "Label_3"


