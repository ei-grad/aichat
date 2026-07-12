#[macro_export]
macro_rules! client_common_fns {
    () => {
        fn global_config(&self) -> &$crate::config::GlobalConfig {
            &self.global_config
        }

        fn extra_config(&self) -> Option<&$crate::client::ExtraConfig> {
            self.config.extra.as_ref()
        }

        fn patch_config(&self) -> Option<&$crate::client::RequestPatch> {
            self.config.patch.as_ref()
        }

        fn name(&self) -> &str {
            Self::name(&self.config)
        }

        fn model(&self) -> &Model {
            &self.model
        }

        fn model_mut(&mut self) -> &mut Model {
            &mut self.model
        }
    };
}

#[macro_export]
macro_rules! impl_client_trait {
    (
        $client:ident,
        ($prepare_chat_completions:path, $chat_completions:path, $chat_completions_streaming:path),
        ($prepare_embeddings:path, $embeddings:path),
        ($prepare_rerank:path, $rerank:path),
    ) => {
        #[async_trait::async_trait]
        impl $crate::client::Client for $crate::client::$client {
            client_common_fns!();

            async fn chat_completions_inner(
                &self,
                client: &reqwest::Client,
                data: $crate::client::ChatCompletionsData,
            ) -> anyhow::Result<$crate::client::ChatCompletionsOutput> {
                let request_data = $prepare_chat_completions(self, data)?;
                let builder = self.request_builder(client, request_data);
                $chat_completions(builder, self.model()).await
            }

            async fn chat_events_inner(
                &self,
                client: &reqwest::Client,
                data: $crate::client::ChatCompletionsData,
            ) -> Result<$crate::client::ChatEventStream> {
                let request_data = $prepare_chat_completions(self, data)?;
                let builder = self.request_builder(client, request_data);
                $chat_completions_streaming(builder, self.model()).await
            }

            async fn embeddings_inner(
                &self,
                client: &reqwest::Client,
                data: &$crate::client::EmbeddingsData,
            ) -> Result<$crate::client::EmbeddingsOutput> {
                let request_data = $prepare_embeddings(self, data)?;
                let builder = self.request_builder(client, request_data);
                $embeddings(builder, self.model()).await
            }

            async fn rerank_inner(
                &self,
                client: &reqwest::Client,
                data: &$crate::client::RerankData,
            ) -> Result<$crate::client::RerankOutput> {
                let request_data = $prepare_rerank(self, data)?;
                let builder = self.request_builder(client, request_data);
                $rerank(builder, self.model()).await
            }
        }
    };
}

#[macro_export]
macro_rules! config_get_fn {
    ($field_name:ident, $fn_name:ident) => {
        $crate::config_get_fn!($field_name, $fn_name, []);
    };
    ($field_name:ident, $fn_name:ident, [$($env_alias:literal),* $(,)?]) => {
        fn $fn_name(&self) -> $crate::client::ConfigFieldResult<String> {
            let env_prefix = Self::name(&self.config);
            $crate::client::resolve_config_field(
                env_prefix,
                stringify!($field_name),
                self.config.$field_name.as_deref(),
                &[$($env_alias),*],
            )
        }
    };
}

#[macro_export]
macro_rules! unsupported_model {
    ($name:expr) => {
        anyhow::bail!("Unsupported model '{}'", $name)
    };
}
