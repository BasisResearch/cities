format: FORCE
	./scripts/clean.sh

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

FORCE: