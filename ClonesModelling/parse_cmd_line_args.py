import argparse
import logging
from math import inf

from .parameters.hypothesis_module_class import get_hypothesis_modules, get_module_names
from .parameters.parameter_class import abbreviate_name


def get_parent_argument_parser(
    resolve_conflicts: bool = False,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=False, conflict_handler="resolve" if resolve_conflicts else "error"
    )

    parser.add_argument(
        "-s", "--seed", type=float, default=2021, help="seed provided to randomiser"
    )
    parser.add_argument(
        "-nc",
        "--initial_basal_cell_number",
        type=int,
        default=500,
        help=("initial number of basal cells in each simulation"),
    )
    parser.add_argument(
        "-sy",
        "--steps_per_year",
        type=int,
        default=365,
        help="number of time increments per simulated year",
    )
    parser.add_argument(
        "-dnp",
        "--dynamic_normalisation_power",
        type=float,
        default=10,
        help=(
            "controls how strongly normalisation tries to preserve initial cell number"
        ),
    )
    parser.add_argument(
        "-nsp",
        "--non_spatial",
        action="store_false",
        dest="spatial",
        help="run simulations without spatial structure",
    )
    parser.add_argument(
        "-ei",
        "--exclude_infants",
        action="store_false",
        dest="include_infants",
        help="only run simulations for patients over 5 years old",
    )
    parser.add_argument(
        "-fpt",
        "--first_patient_test",
        action="store_true",
        help="only run first patient, for a quick test run",
    )
    parser.add_argument(
        "-crd",
        "--cell_records_directory",
        type=str,
        default=".",
        help="parent directory for the cell_records folder",
    )
    parser.add_argument(
        "-ld",
        "--logging_directory",
        type=str,
        default="logs",
        help="folder to use for logs, for cluster use primarily",
    )
    parser.add_argument(
        "-rf",
        "--record_frequency",
        type=int,
        default=3,
        help="frequency of recording data (records per year)",
    )

    for module_name in get_module_names():
        parser.add_argument(
            f"-{abbreviate_name(module_name)}",
            f"--{module_name}_module",
            action="store_true",
            help=f"activate {module_name} module",
        )

    parser.add_argument(
        "-eng",
        "--exclude_nature_genetics",
        action="store_true",
        help="only use Nature 2020 Clones data",
    )
    parser.add_argument(
        "-ls",
        "--linear_scale_comparison",
        action="store_true",
        help="compare distributions of mutational burden on a linear (not log) scale",
    )
    parser.add_argument(
        "-iss",
        "--ignore_smoking_signature_mutations",
        action="store_false",
        dest="compare_smoking_signature_mutations",
        help="compare mutations by smoking signature using 2D wasserstein distance",
    )
    parser.add_argument(
        "-wl",
        "--weight_loss_by_cell_count",
        action="store_true",
        help="weight loss values by the number of cells sampled from the patient",
    )
    parser.add_argument(
        "-zpp",
        "--zero_population_penalisation_factor",
        type=float,
        default=3.0,
        help="degree of penalisation for zero population errors",
    )
    parser.add_argument(
        "-rfc",
        "--remove_fitness_change",
        action="store_true",
        help="remove fitness change upon mutation from the model",
    )
    parser.add_argument(
        "-vmr",
        "--vary_mutation_rate",
        action="store_true",
        help="allow mutations to increase mutation rate",
    )

    parser.add_argument(
        "-py",
        "--python",
        action="store_false",
        dest="julia",
        help="run simulations in python rather than julia",
    )
    return parser


def get_single_run_argument_parser() -> argparse.ArgumentParser:
    parent_parser = get_parent_argument_parser()
    parser = argparse.ArgumentParser(parents=[parent_parser])

    parser.add_argument(
        "-rid", "--run_id", type=str, default="local", help="id for log file names"
    )
    parser.add_argument(
        "-cl",
        "--console_logging_level",
        type=int,
        default=1,
        help=(
            "level of logging printed to console; int 0-4, selected from "
            "(0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL)"
        ),
    )
    parser.add_argument(
        "-fl",
        "--file_logging_level",
        default=0,
        type=int,
        help=(
            "level logged in file; int 0-4; selecting from "
            "(0:DEBUG, 1:INFO, 2:WARNING, 3:ERROR, 4:CRITICAL)"
        ),
    )
    parser.add_argument(
        "-rgd",
        "--record_graph_data",
        action="store_true",
        help="perform slower run to record data for plots",
    )

    for hypothesis_module in get_hypothesis_modules(spatial=True):
        hypothesis_module.add_parameter_arguments(parser)
    return parser


def parse_single_run_arguments() -> argparse.Namespace:
    parser = get_single_run_argument_parser()
    parsed_args = parser.parse_args()

    convert_logging_levels(parsed_args)
    ensure_parameter_args(parsed_args)

    if not parsed_args.record_graph_data:
        parsed_args.record_frequency = 0
    elif parsed_args.record_frequency < 1:
        print("WARNING: record frequency must be at least once/year, setting to 1")
        parsed_args.record_frequency = 1

    return parsed_args


def parse_compare_modules_arguments() -> argparse.Namespace:
    parser = get_single_run_argument_parser()
    parser.add_argument(
        "-np",
        "--number_of_processes",
        type=int,
        default=4,
        help="number of processes to run in parallel",
    )
    parser.add_argument(
        "-ns",
        "--number_of_seeds",
        type=int,
        default=3,
        help="number of seeds to run for each implementation",
    )
    parser.add_argument(
        "-hpc",
        "--high_performance_cluster",
        action="store_true",
        help="parallelise via high performance cluster rather than multiprocessing",
    )
    parsed_args = parser.parse_args()

    convert_logging_levels(parsed_args)
    return parsed_args


def get_synthetic_data_test_arguments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-np",
        "--number_of_processes",
        type=int,
        default=4,
        help="number of processes to run in parallel for each ABC run",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2021,
        help="random seed for generating synthetic data and ABC seeds",
    )
    parser.add_argument(
        "-ns",
        "--number_of_seeds",
        type=int,
        default=3,
        help="number of seeds to run for each implementation",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="run on local machine rather than cluster",
    )
    parser.add_argument(
        "-aa",
        "--abc_arguments",
        type=str,
        help="arguments to pass to abc script, in quotes",
    )
    return parser


def ensure_parameter_args(parsed_args):
    hypothesis_modules = get_hypothesis_modules(spatial=True)
    # ensure if any module's arguments are set then that module is activated
    for hypothesis_module in hypothesis_modules:
        if hypothesis_module.name == "base":
            continue
        for parameter in hypothesis_module.parameters:
            if (
                not getattr(parsed_args, hypothesis_module.name + "_module")
                and getattr(parsed_args, parameter.name) is not None
            ):
                setattr(parsed_args, hypothesis_module.name + "_module", True)
    # ensure if any module is activated then that module's arguments are set
    for hypothesis_module in hypothesis_modules:
        for parameter in hypothesis_module.parameters:
            if getattr(parsed_args, parameter.name) is None:
                if hypothesis_module.name == "base" or getattr(
                    parsed_args, hypothesis_module.name + "_module"
                ):
                    print(
                        f"WARNING: {parameter.name} not set, but "
                        f"{hypothesis_module.name} module is active; "
                        f"setting to default value {parameter.default_value}"
                    )
                    setattr(parsed_args, parameter.name, parameter.default_value)
                else:
                    setattr(
                        parsed_args,
                        parameter.name,
                        parameter.inactive_value,
                    )


def convert_logging_levels(parsed_args):
    logging_levels = (
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    )
    parsed_args.file_logging_level, parsed_args.console_logging_level = (
        logging_levels[parsed_args.file_logging_level],
        logging_levels[parsed_args.console_logging_level],
    )


def get_plot_mutational_burden_arguments_parser() -> argparse.ArgumentParser:
    parser = get_parent_argument_parser()

    parser.add_argument(
        "-rid",
        "--run_id",
        type=str,
        default="local",
        help="id of simulation run to plot",
    )
    parser.add_argument(
        "-bw",
        "--bin_width",
        type=int,
        default=10,
        help="width of bins for mutation burden histogram",
    )
    return parser


def parse_plot_mutational_burden_arguments() -> argparse.Namespace:
    parser = get_plot_mutational_burden_arguments_parser()
    return parser.parse_args()


def get_abc_arguments_parser(resolve_conflicts=False) -> argparse.ArgumentParser:
    parser = get_parent_argument_parser(resolve_conflicts)
    # resolve conflicts for abc_batch.py to allow multiple values for some args

    parser.add_argument(
        "-rid", "--run_id", type=str, default="local", help="id for log file names"
    )
    parser.add_argument(
        "-np",
        "--number_of_processes",
        type=int,
        help="number of processes to create in parallel",
    )
    parser.add_argument(
        "-aps",
        "--abc_population_size",
        type=int,
        default=2,
        help="number of particles in ABC-SMC run",
    )
    parser.add_argument(
        "-me",
        "--minimum_epsilon",
        type=float,
        default=0.0,
        help="minimum epsilon in the ABC run",
    )
    parser.add_argument(
        "-mp",
        "--max_populations",
        type=int,
        default=inf,
        help="number of populations to terminate after, regardless of success",
    )
    parser.add_argument(
        "-ms",
        "--max_simulations",
        type=int,
        default=inf,
        help="total number of simulations to terminate after, regardless of success",
    )
    parser.add_argument(
        "-mw",
        "--max_walltime",
        type=float,
        help="maximum walltime in hours",
    )
    parser.add_argument(
        "-ma",
        "--min_acceptance_rate",
        type=float,
        default=0.0,
        help="minimum acceptance rate for ABC run",
    )
    parser.add_argument(
        "-med",
        "--minimum_epsilon_difference",
        type=float,
        default=0.0,
        help="minimum difference between epsilons for ABC run, after which to stop",
    )
    parser.add_argument(
        "-ad", "--adaptive_distance", action="store_true", help="use adaptive distance"
    )
    parser.add_argument(
        "-fdr",
        "--fix_division_rate",
        action="store_false",
        dest="vary_division_rate",
        help="fix values for division rate parameters",
    )
    parser.add_argument(
        "-fp",
        "--fix_paradigm",
        action="store_true",
        help="fix paradigm rather than choosing which is best out of all sub-paradigms",
    )
    parser.add_argument(
        "-scs",
        "--single_core_sampler",
        action="store_true",
        help="use single core sampler rather than parallelised sampler for debugging",
    )
    parser.add_argument(
        "-bsn",
        "--bootstrap_sample_number",
        type=int,
        default=1000,
        help="number of bootstrap samples to run",
    )
    parser.add_argument(
        "-psp",
        "--posterior_samples_per_paradigm",
        type=int,
        default=5,
        help="number of samples to take from posterior for each surviving paradigm",
    )
    parser.add_argument(
        "-prc",
        "--posterior_replicate_count",
        type=int,
        default=2,
        help="number of replicates to run of each posterior simulation",
    )
    parser.add_argument(
        "-mm",
        "--max_modules",
        type=int,
        default=2,
        help="maximum number of modules to include in a paradigm, excluding base",
    )
    parser.add_argument(
        "-ewp",
        "--evenly_weighted_paradigms",
        action="store_false",
        dest="weight_paradigms",
        help="weight paradigms equally rather than inversely by number of modules",
    )
    parser.add_argument(
        "-sc",
        "--replicate_count",
        type=int,
        default=2,
        help="number of simulations to run for each ABC probe",
    )
    parser.add_argument(
        "-ssc",
        "--subsample_count",
        type=int,
        default=100,
        help="number of subsamples to take from each simulation",
    )
    parser.add_argument(
        "-tssc",
        "--tree_subsample_count",
        type=int,
        default=10,
        help="number of subsamples to take from each tree",
    )
    parser.add_argument(
        "-sap",
        "--skip_animated_plots",
        action="store_true",
        help="skip time-consuming animated plots for posterior samples",
    )
    for hypothesis_module in get_hypothesis_modules(spatial=True):
        hypothesis_module.add_parameter_arguments(parser)
    return parser


def get_abc_length_constraints(
    parsed_abc_args: argparse.Namespace,
) -> list[tuple[str, bool]]:
    constraints_with_defaults: list[tuple[str, float | None]] = [
        ("minimum_epsilon", 0.0),
        ("max_populations", inf),
        ("max_simulations", inf),
        ("max_walltime", None),
        ("min_acceptance_rate", 0.0),
        ("minimum_epsilon_difference", 0.0),
    ]
    return [
        (constraint_name, getattr(parsed_abc_args, constraint_name) == default_value)
        for constraint_name, default_value in constraints_with_defaults
    ]


def parse_abc_arguments() -> argparse.Namespace:
    parsed_args = get_abc_arguments_parser().parse_args()
    ensure_termination_condition(parsed_args)
    if (
        parsed_args.number_of_processes
        and parsed_args.number_of_processes > 1
        and parsed_args.single_core_sampler
    ):
        raise ValueError("Cannot use single core sampler with multiple processes")
    if not parsed_args.fix_paradigm:
        for module in get_module_names():
            setattr(parsed_args, module + "_module", True)
    return parsed_args


def ensure_termination_condition(parsed_args):
    if all(
        parameter_is_default
        for _, parameter_is_default in get_abc_length_constraints(parsed_args)
    ):
        raise ValueError("No termination condition specified; use -h for help")


def get_abc_batch_arguments_parser() -> argparse.ArgumentParser:
    parser = get_abc_arguments_parser(resolve_conflicts=True)

    parser.set_defaults(
        logging_directory="/SAN/medic/hselway_omics/abc",
        number_of_processes=2,
        run_id=None,
    )

    parser.add_argument(
        "-nr",
        "--number_of_repetitions",
        type=int,
        default=1,
        help="number of repeats of each ABC run to test consistency of results",
    )
    # override abc arguments to allow multiple values, for testing each
    parser.add_argument(
        "-aps",
        "--abc_population_size",
        type=int,
        nargs="+",
        default=[20, 40],
        help="population sizes to use for each ABC run",
    )
    return parser


def parse_abc_batch_args():
    parsed_args = get_abc_batch_arguments_parser().parse_args()
    ensure_termination_condition(parsed_args)
    if not parsed_args.fix_paradigm:
        # set all modules to True
        for module in get_module_names():
            setattr(parsed_args, module + "_module", True)
    return parsed_args


def get_reprocess_abc_output_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-rid", "--run_id", type=str, default="local", help="id of run to plot"
    )
    parser.add_argument(
        "-ld",
        "--logging_directory",
        type=str,
        default="/SAN/medic/hselway_omics/abc",
        help="directory in which to find pickled history file",
    )
    parser.add_argument(
        "-bsn",
        "--bootstrap_sample_number",
        type=int,
        default=1000,
        help="number of bootstrap samples to run",
    )
    return parser


def parse_reprocess_abc_output_arguments() -> argparse.Namespace:
    return get_reprocess_abc_output_argument_parser().parse_args()


def get_id_test_argument_parser() -> argparse.ArgumentParser:
    parser = get_parent_argument_parser()
    parser.add_argument(
        "-spp",
        "--simulations_per_paradigm",
        type=int,
        default=100,
        help="number of simulations to run for each paradigm",
    )
    parser.add_argument(
        "-rc",
        "--replicate_count",
        type=int,
        default=10,
        help="number of times to run each simulation",
    )
    parser.add_argument(
        "-np",
        "--number_of_processes",
        type=int,
        default=4,
        help="number of processes to run in parallel",
    )
    parser.add_argument(
        "-hpc",
        "--high_performance_cluster",
        action="store_true",
        help="run on cluster rather than locally",
    )
    parser.add_argument("-rid", "--run_id", type=str)

    for hypothesis_module in get_hypothesis_modules(spatial=True):
        hypothesis_module.add_parameter_arguments(parser)
    # args from abc to allow for hijacking the priors
    parser.add_argument(
        "-mm",
        "--max_modules",
        type=int,
        default=2,
        help="maximum number of modules to include in a paradigm, excluding base",
    )
    parser.add_argument(
        "-fp",
        "--fix_paradigm",
        action="store_true",
        help="fix paradigm rather than choosing which is best out of all sub-paradigms",
    )
    parser.add_argument(
        "-fdr",
        "--fix_division_rate",
        action="store_false",
        dest="vary_division_rate",
        help="fix values for division rate parameters",
    )
    parser.add_argument(
        "-spc",
        "--supersample_patient_cohort",
        action="store_true",
        help="Use larger synthetic patient cohort",
    )
    return parser


def parse_id_test_arguments() -> argparse.Namespace:
    parsed_args = get_id_test_argument_parser().parse_args()
    if not parsed_args.fix_paradigm:
        for module in get_module_names():
            setattr(parsed_args, module + "_module", True)
    if parsed_args.high_performance_cluster:
        parsed_args.logging_directory = (
            "/cluster/project2/clones_modelling/identifiability_test"
        )
    return parsed_args


def get_id_test_comparison_arguments_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ld",
        "--logging_directory",
        type=str,
        default="logs",
        help="directory in which to find pickled history file",
    )
    parser.add_argument(
        "-rid", "--run_id", type=str, default="local", help="id for log file names"
    )
    parser.add_argument(
        "-nps",
        "--n_pairs_to_sample",
        type=int,
        help="number of pairs to sample for each similarity level",
    )
    parser.add_argument(
        "-hpc",
        "--high_performance_cluster",
        action="store_true",
        help="run on cluster rather than locally",
    )
    parser.add_argument(
        "-sl",
        "--similarity_level",
        type=int,
        default=0,
        help=(
            "similarity level to compare (0: replicate, 1: intra-paradigm, "
            "2: inter-paradigm)"
        ),
    )
    parser.add_argument(
        "-iss",
        "--ignore_smoking_signature_mutations",
        action="store_false",
        dest="compare_smoking_signature_mutations",
        help="compare mutations by smoking signature using 2D wasserstein distance",
    )
    parser.add_argument(
        "-hold_jid",
        type=int,
        help="job id to hold on before starting comparison job",
    )
    parser.add_argument(
        "-gn",
        "--group_number",
        type=int,
        help="group number to compare",
    )
    return parser


def parse_id_test_comparison_arguments() -> argparse.Namespace:
    parsed_args = get_id_test_comparison_arguments_parser().parse_args()
    if parsed_args.high_performance_cluster:
        parsed_args.logging_directory = (
            "/cluster/project2/clones_modelling/identifiability_test"
        )
    return parsed_args
