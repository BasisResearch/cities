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

done: FORCE
	./scripts/clean.sh
	./scripts/lint.sh
	./scripts/test.sh
	./scripts/test_notebooks.sh

FORCE: