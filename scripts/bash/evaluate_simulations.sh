#!/bin/bash
set -euo pipefail

# Configurable arrays
# run_names=(
#     "2025may23_oldmodel_HLR" "2025may23_oldmodel" "2025may23_finetuned_new" "2025may23_finetuned_newHLR"
#     "2025may23_finetuned_HLR" "2025may23_finetuned" "SwinUnet2025May23HLR" "SwinUnet2025May23"
#     "Unet28May2025" "Unet25May2025HLR"
# )
# model_names=(
#     "UnetResnet34Tr" "UnetResnet34Tr" "UnetResnet34Tr" "UnetResnet34Tr" "UnetResnet34Tr"
#     "UnetResnet34Tr" "SwinUnet" "SwinUnet" "Unet" "Unet"
# )

run_names=(
    "Unet28May2025" "Unet25May2025HLR"
)
model_names=(
    "Unet" "Unet"
)

# Check arrays have the same length
if [ "${#run_names[@]}" -ne "${#model_names[@]}" ]; then
    echo "Error: The number of run names must equal the number of model names."
    exit 1
fi

# Common settings
TEST_DATASET="/mnt/g/data/PhD Projects/SR/120deg2_shark_sides/Test"
BATCH_SIZE=36
INPUT_CLASSES='["24", "250", "350", "500"]'
TARGET_CLASSES='["500SR"]'

# Directories and scripts
OUTPUT_DIR="/mnt/d/SPIRE-SR-AI/configs"   # not used here but kept as reference
CATALOG_SCRIPT="/mnt/d/SPIRE-SR-AI/scripts/evaluate/get_SR_target_catalog.py"
EVALUATION_SCRIPT="/mnt/d/SPIRE-SR-AI/scripts/evaluate/evaluate.py"

# Function to generate config file
generate_config() {
    local file="$1"
    local model="$2"
    local run_name="$3"
    local data_section="$4"

    cat <<EOF > "$file"
model:
    # Model specific settings, do not change!
    model: "${model}"        # Must match the configuration folder
    run_name: "${run_name}"  # Run identifier (e.g., start date)
    input_shape: [256, 256, 4]
    output_shape: [256, 256, 1]

sys_config:
    n_cpu_cores: 14

data:
${data_section}
EOF
}

for ((i = 0; i < ${#run_names[@]}; i++)); do
    RUN_NAME="${run_names[$i]}"
    MODEL_NAME="${model_names[$i]}"
    CONFIG_FILE="temp_config_${RUN_NAME}.yaml"
    TARGET_DIR="/mnt/d/SPIRE-SR-AI/data/raw/catalogs/sim"
    
    echo "Processing run: ${RUN_NAME} with model: ${MODEL_NAME}"
        
    # Data section for SR target catalog preparation
    if [ "$i" -eq 0 ]; then
        target_cat_line="target_catalog_output_dir: \"${TARGET_DIR}\""
    else
        target_cat_line="target_catalog_output_dir: null"
    fi
    catalog_data="    test_dataset_path: \"${TEST_DATASET}\"
    ${target_cat_line}
    test_batch_size: ${BATCH_SIZE}
    input: ${INPUT_CLASSES}
    target: ${TARGET_CLASSES}"

    generate_config "$CONFIG_FILE" "${MODEL_NAME}" "${RUN_NAME}" "${catalog_data}"

    # Generate SR catalogs
    python3 "$CATALOG_SCRIPT" --config "$CONFIG_FILE"

    echo "Evaluating model: ${MODEL_NAME} with run name: ${RUN_NAME}"

    # Data section for evaluation
    evaluation_data="    test_dataset_path: \"${TEST_DATASET}\"
    input_catalog_path: \"/mnt/d/SPIRE-SR-AI/data/raw/catalogs/sim/120_deg2_shark_sides_input_test_catalog.fits\"
    native_catalog_path: \"/mnt/d/SPIRE-SR-AI/data/raw/catalogs/sim/SPIRE500_native_catalog.fits\"
    target_catalog_path: \"/mnt/d/SPIRE-SR-AI/data/raw/catalogs/sim/500SR_target_catalog.fits\"
    test_batch_size: ${BATCH_SIZE}
    input: ${INPUT_CLASSES}
    target: ${TARGET_CLASSES}"

    generate_config "$CONFIG_FILE" "${MODEL_NAME}" "${RUN_NAME}" "${evaluation_data}"

    # Run evaluation
    python3 "$EVALUATION_SCRIPT" --config "$CONFIG_FILE"

    rm "$CONFIG_FILE"
done