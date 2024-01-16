import os
os.system("pip install hyperpyyaml")
from hyperpyyaml.core import recursive_update, _walk_tree_and_resolve
from datetime import date
from io import StringIO
import ruamel.yaml
import inspect
import os
import shutil
import logging

logger = logging.getLogger(__name__)


def create_experiment_directory(
        experiment_directory,
        hyperparams_to_save=None,
        overrides={},
):
    """Create the output folder and relevant experimental files.

    Arguments
    ---------
    experiment_directory : str
        The place where the experiment directory should be created.
    hyperparams_to_save : str
        A filename of a yaml file representing the parameters for this
        experiment. If passed, references are resolved, and the result is
        written to a file in the experiment directory called "hyperparams.yaml".
    overrides : dict
        A mapping of replacements made in the yaml file, to save in yaml.
    log_config : str
        A yaml filename containing configuration options for the logger.
    save_env_desc : bool
        If True, an environment state description is saved to the experiment
        directory, in a file called env.log in the experiment directory.
    """
    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)

        # Write the parameters file
        if hyperparams_to_save is not None:
            hyperparams_filename = os.path.join(
                experiment_directory, "hyperparams.yaml"
            )
            with open(hyperparams_to_save) as f:
                resolved_yaml = resolve_references(f, overrides)
            with open(hyperparams_filename, "w") as w:
                print("# Generated %s from:" % date.today(), file=w)
                print("# %s" % os.path.abspath(hyperparams_to_save), file=w)
                print("# yamllint disable", file=w)
                shutil.copyfileobj(resolved_yaml, w)

        # Copy executing file to output directory
        module = inspect.getmodule(inspect.currentframe().f_back)
        if module is not None:
            callingfile = os.path.realpath(module.__file__)
            shutil.copy(callingfile, experiment_directory)

        #Log beginning of experiment!
        logger.info("Beginning experiment!")
        logger.info(f"Experiment folder: {experiment_directory}")

def resolve_references(yaml_stream, overrides=None, overrides_must_match=False):
    r'''Resolves inter-document references, a component of HyperPyYAML.

    Arguments
    ---------
    yaml_stream : stream
        A file-like object or string with the contents of a yaml file
        written with the HyperPyYAML syntax.
    overrides : mapping or str
        Replacement values, either in a yaml-formatted string or a dict.
    overrides_must_match : bool
        Whether an error will be thrown when an override does not match
        a corresponding key in the yaml_stream. This is the opposite
        default from ``load_hyperpyyaml`` because ``resolve_references``
        doesn't need to be as strict by default.

    Returns
    -------
    stream
        A yaml-formatted stream with all references and overrides resolved.
    '''
    # find imported yaml location relative to main yaml file
    file_path = None
    if hasattr(yaml_stream, "name"):
        file_path = os.path.dirname(os.path.realpath(yaml_stream.name))

    # Load once to store references and apply overrides
    # using ruamel.yaml to preserve the tags
    ruamel_yaml = ruamel.yaml.YAML()
    preview = ruamel_yaml.load(yaml_stream)

    if overrides is not None and overrides != "":
        if isinstance(overrides, str):
            overrides = ruamel_yaml.load(overrides)
        recursive_update(preview, overrides, must_match=overrides_must_match)
    _walk_tree_and_resolve("root", preview, preview, overrides, file_path)

    # Dump back to string so we can load with bells and whistles
    yaml_stream = StringIO()
    ruamel_yaml.dump(preview, yaml_stream)
    yaml_stream.seek(0)

    return yaml_stream

