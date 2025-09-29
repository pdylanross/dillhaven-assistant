use anyhow::Result;
use hf_hub::api::sync::{Api as SyncApi, ApiBuilder as SyncApiBuilder, ApiRepo as SyncApiRepo};
use hf_hub::api::tokio::{
    Api as AsyncApi, ApiBuilder as AsyncApiBuilder, ApiRepo as AsyncApiRepo, ApiRepo,
};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct HuggingFaceApiManger {
    async_api: Arc<AsyncApi>,
    sync_api: Arc<SyncApi>,
}

impl HuggingFaceApiManger {
    pub async fn new() -> Result<Self> {
        let async_api = Arc::new(AsyncApiBuilder::from_env().build()?);
        let sync_api = Arc::new(SyncApiBuilder::from_env().build()?);

        Ok(Self {
            async_api,
            sync_api,
        })
    }

    pub fn get_async_api(&self) -> Arc<AsyncApi> {
        self.async_api.clone()
    }

    pub fn get_sync_api(&self) -> Arc<SyncApi> {
        self.sync_api.clone()
    }
}

pub async fn load_json_file_as_async<T: DeserializeOwned>(
    repo: &AsyncApiRepo,
    path: &str,
) -> Result<T> {
    let json_file = repo.get(path).await?;
    des_json_file(json_file)
}

pub async fn load_json_file_as_sync<T: DeserializeOwned>(
    repo: &SyncApiRepo,
    path: &str,
) -> Result<T> {
    let json_file = repo.get(path)?;
    des_json_file(json_file)
}

fn des_json_file<T: DeserializeOwned>(path: PathBuf) -> Result<T> {
    let file = File::open(path)?;
    Ok(serde_json::from_reader(file)?)
}

#[derive(Debug, Clone, Deserialize)]
pub struct SafeTensorsIndex {
    pub metadata: serde_json::Value,
    pub weight_map: HashMap<String, String>,
}

impl SafeTensorsIndex {
    pub async fn load_tensors(&self, api: &ApiRepo) -> Result<Vec<PathBuf>> {
        let mut safetensors_files = HashSet::new();
        for v in self.weight_map.values() {
            safetensors_files.insert(v.to_string());
        }

        let mut ret = Vec::new();
        for v in safetensors_files {
            let f = api.get(&v).await?;
            ret.push(f);
        }

        Ok(ret)
    }
}
