# Use a small, Conda-ready base
FROM condaforge/mambaforge:24.7.1-0

# Copy your code into the image
WORKDIR /app
COPY pyproject.toml README.md LICENSE environment.yml ./
COPY src/ ./src

# entrypoint script (for activation of conda environment)
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create the environment
RUN mamba env create -f environment.yml \
    && mamba clean --all --yes \
    && rm environment.yml

# Make sure the conda env is activated in all RUN/CMD
ENTRYPOINT ["/entrypoint.sh"]

# By default run your console script
CMD ["mbiomekit", "--help"]
