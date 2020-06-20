set -gx VULKAN_SDK (pwd)"/vulkan-sdk-1.2.141.2/macOS";
set -x -p PATH "$VULKAN_SDK/bin";
set -x DYLD_LIBRARY_PATH "$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH";
set -x VK_LAYER_PATH "$VULKAN_SDK/share/vulkan/explicit_layer.d";
set -x VK_ICD_FILENAMES "$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json";
set -x SHADERC_LIB_DIR (pwd)"/shaderc/lib";
