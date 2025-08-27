using FastSimulation
using PackageCompiler

sys_image_directory =
    FastSimulation.ParseCmdLineArgs.parse_cmd_line_args()["this_probe_logging_directory"]

create_sysimage(
    ["FastSimulation"],
    sysimage_path = "$sys_image_directory/sysimage.so",
    precompile_execution_file = "ClonesModelling/FastSimulation/precompile.jl",
)
