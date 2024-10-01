format: FORCE
	./scripts/clean.sh


path ?= .

format_path: FORCE
	./scripts/clean_path.sh $(path)

lint: FORCE
	./scripts/lint.sh

test: FORCE
	./scripts/test.sh

test_all: FORCE
	./scripts/clean.sh
	./scripts/lint.sh
	./scripts/test.sh
	./scripts/test_notebooks.sh

test_notebooks: FORCE
	./scripts/test_notebooks.sh

api/requirements.txt: FORCE
	pip-compile --extra api --output-file api/requirements.txt

api-container-build: FORCE
	mkdir -p build
	#cd build && python ../cities/deployment/tracts_minneapolis/train_model.py
	cp -r cities build
	cp -r api/* build
	cp .env build
	cd build && docker build --platform linux/amd64 -t cities-api . 

api-container-push:
	docker tag cities-api us-east1-docker.pkg.dev/cities-429602/cities/cities-api
	docker push us-east1-docker.pkg.dev/cities-429602/cities/cities-api

run-api-local:
	sudo -E docker run --rm -it -e PORT=8081 -e PASSWORD -p 3001:8081 cities-api

FORCE:
