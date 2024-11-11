# caise2025data
Data analyzed for the CAiSE 2025 paper

## How to Execute the Code

You can run the code in your project by executing the `main.py` script. The script can be executed in two different ways: with a user-provided catalog path as an argument or by using the default catalog path.

### Execution Options

To execute the code, use the command line as follows:

```bash
usage: main.py [-h] [catalog_path]

Receives OntoUML models path.

positional arguments:
  catalog_path  Path to the input data source directory. Defaults to '../ontouml-models' if not provided.

options:
  -h, --help    show this help message and exit
```

#### 1. Using the Default Catalog Path

If you do not provide any arguments, the script will use `../ontouml-models` as the default catalog path:

```bash
python main.py
```

- **Default Path**: The script uses the default catalog path defined in the configuration.
- **Use Case**: This is suitable if your OntoUML models are located in the default directory.

#### 2. Providing a Custom Catalog Path

You can specify a custom path to your OntoUML models as a command-line argument:

```bash
python main.py "/path/to/your/models"
```

- **Custom Path**: Replace `"/path/to/your/models"` with the actual path to your OntoUML models. Quotes are required if the path contains spaces or special characters.
- **Use Case**: Use this option when your models are located in a different directory than the default one.

### Notes

- Make sure to use quotes around the path if it contains spaces or special characters.
- Use the `-h` or `--help` option to display the help message for more details on how to use the script.
