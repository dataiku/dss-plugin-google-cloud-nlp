# Public variable to be set by the user in the Makefile
# TODO validate that version has valid syntax
TARGET_DSS_VERSION=7.0

# Private variables to be set by the user in the environment
ifndef DKU_PLUGIN_DEVELOPER_ORG
$(error the DKU_PLUGIN_DEVELOPER_ORG environment variable is not set)
endif
ifndef DKU_PLUGIN_DEVELOPER_TOKEN
$(error the DKU_PLUGIN_DEVELOPER_TOKEN environment variable is not set)
endif
ifndef DKU_PLUGIN_ARTIFACT_REPO_URL
$(error the DKU_PLUGIN_ARTIFACT_REPO_URL environment variable is not set)
endif

# evaluate additional variable
plugin_id=`cat plugin.json | python -c "import sys, json; print(json.load(sys.stdin)['id'])"`
plugin_version=`cat plugin.json | python -c "import sys, json; print(json.load(sys.stdin)['version'])"`
archive_file_name="dss-plugin_${plugin_id}_${plugin_version}.zip"
# TODO check for case where DKU_PLUGIN_ARTIFACT_REPO_URL ends with /
# TODO check that no variables contains /
artifact_repo_target="${DKU_PLUGIN_DEVELOPER_REPO_URL}/${TARGET_DSS_VERSION}/${DKU_PLUGIN_DEVELOPER_ORG}/${plugin_id}/${plugin_version}/${archive_file_name}"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`


plugin:
	@echo "[START] Archiving plugin to dist/ folder..."
	@cat plugin.json | json_pp > /dev/null
	@rm -rf dist
	@mkdir dist
	@echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json
	@git archive -v -9 --format zip -o dist/${archive_file_name} HEAD
	@zip -u dist/${archive_file_name} release_info.json
	@rm release_info.json
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

publish: plugin
	@echo "[START] Publishing archive to artifact repository..."
	@curl -H "Authorization: Bearer ${DKU_PLUGIN_DEVELOPER_TOKEN}>" -X PUT ${artifact_repo_target} -T dist/${archive_file_name}
	@echo "[SUCCESS] Publishing archive to artifact repository: Done!"


dist-clean:
	rm -rf dist

