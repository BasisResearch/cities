format: FORCE
	./scripts/clean.sh


path ?= .

format_path: FORCE
	./scripts/clean_path.sh $(path)

lint: FORCE
	./scripts/lint.sh

test: FORCE
	./scripts/lint.sh
	./scripts/test.sh
	./scripts/test_notebooks.sh

test_notebooks: FORCE
	./scripts/test_notebooks.sh

done: FORCE
	./scripts/clean.sh
	./scripts/lint.sh
	./scripts/test.sh
	./scripts/test_notebooks.sh

api/requirements.txt: FORCE
	pip-compile --extra api --output-file api/requirements.txt

build/tracts_model_params.pth build/tracts_model_guide.pkl: FORCE
	mkdir -p build
	cd build && python ../cities/deployment/tracts_minneapolis/train_model.py

api-container: FORCE api/requirements.txt build/tracts_model_params.pth build/tracts_model_guide.pkl
	mkdir -p build
	cp -r cities build
	cp -r api/ build
	cd build && docker build -t BasisResearch/cities-api .

FORCE:
