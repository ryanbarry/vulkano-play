#!/usr/bin/env sh

# download sdk from LunarG: https://vulkan.lunarg.com/sdk/home
# when i started, the url below was the latest version

wget https://sdk.lunarg.com/sdk/download/1.2.141.2/mac/vulkansdk-macos-1.2.141.2.dmg
hdiutil attach vulkansdk-macos-1.2.141.2.dmg
mkdir vulkan-sdk-1.2.141.2
cp -r /Volumes/vulkansdk-macos-1.2.141.2/* vulkan-sdk-1.2.141.2/
hdiutil detach /Volumes/vulkansdk-macos-1.2.141.2


# download shaderc, so we have libshaderc: https://github.com/google/shaderc#downloads
# (seems like the vulkan sdk actually includes this, so maybe it isn't needed?)

wget -O shaderc-macos-20200616.tgz https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/macos/continuous_clang_release/317/20200616-232648/install.tgz
mkdir shaderc
tar xzf shaderc-macos-20200616.tgz -C shaderc/ --strip-components 1

echo "deps have been downloaded, now just use the ./setup-env.fish before you cargo build/run"
