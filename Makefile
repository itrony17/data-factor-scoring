.PHONY: install lint test demo score report publish-dry-run docker-build clean

install:            ## pip install with dev dependencies
	pip install -e ".[dev]"

lint:               ## ruff static checks
	ruff check src tests

test:               ## full pytest suite (offline, no API keys)
	pytest -v

demo:               ## generate demo/quality_demo.db + demo config
	dqfs demo

score: demo         ## full scoring run on the demo database
	dqfs score --config demo/dqfs-demo.yaml \
		--dsn sqlite:///demo/quality_demo.db \
		--table customers --target churn --dataset-name customers_demo

report:             ## re-render artifacts from results/latest.json
	dqfs report

publish-dry-run:    ## preview what `dqfs publish` would commit
	dqfs publish --repo $(REPO) --branch gh-results --dry-run

docker-build:       ## build the container image
	docker build -t dqfs:latest .

clean:              ## remove generated artifacts
	rm -rf results demo/*.db .pytest_cache .ruff_cache build dist *.egg-info
