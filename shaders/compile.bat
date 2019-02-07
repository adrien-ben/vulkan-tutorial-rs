del shader.vert.spv
del shader.frag.spv

%VULKAN_SDK%/Bin32/glslangValidator.exe -V shader.vert
%VULKAN_SDK%/Bin32/glslangValidator.exe -V shader.frag

ren vert.spv shader.vert.spv
ren frag.spv shader.frag.spv
