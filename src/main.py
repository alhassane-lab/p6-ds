from src.pipelines.preprocess import process_data
from src.pipelines.model import extract_features
from src.utils.envconf import get_var_envs
import click


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """
    Root command.
    """
    ctx.ensure_object(dict)
    ctx.obj["file_name"] = get_var_envs()['file']


main.add_command(process_data)
main.add_command(extract_features)

if __name__ == "__main__":
    main()
