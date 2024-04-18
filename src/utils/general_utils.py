import yaml


def safe_load_yaml(yaml_path: str):
    """
    Closes the yaml if there's a parsing issue when reading into a dict
    Raises the error, no sense continuing execution
    """

    try:
        with open(yaml_path) as file:
            param_dict = yaml.load(file, Loader=yaml.FullLoader)
            return param_dict

    except BaseException as e:

        if 'file' in locals():
            file.close()
        raise e
