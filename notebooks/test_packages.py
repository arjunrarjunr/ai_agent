import pkg_resources

packages = [
    "numpy", "pandas", "jupyterlab", "statsmodels", "sqlalchemy",
    "mysql-connector-python", "seaborn", "matplotlib", "langchain-ollama", "streamlit"
]

for package in packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"{package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"{package}: not installed")