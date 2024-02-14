//! This is an translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::ffi::CStr;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::LlamaToken;

#[derive(clap::Parser)]
struct Args {
    /// The path to the model
    model_path: PathBuf,
    /// The prompt
    #[clap(default_value = "Hello my name is")]
    prompt: String,
    /// Disable offloading layers to the gpu
    #[cfg(feature = "cublas")]
    #[clap(long)]
    disable_gpu: bool,
}

fn main() -> Result<()> {
    let params = Args::parse();

    // init LLM
    let backend = LlamaBackend::init()?;

    // offload all layers to the gpu
    let model_params = {
        #[cfg(feature = "cublas")]
        if !params.disable_gpu {
            LlamaModelParams::default().with_n_gpu_layers(1000)
        } else {
            LlamaModelParams::default()
        }
        #[cfg(not(feature = "cublas"))]
        LlamaModelParams::default()
    };

    let model = LlamaModel::load_from_file(&backend, params.model_path, &model_params)
        .with_context(|| "unable to load model")?;

    // initialize the context
    let ctx_params = LlamaContextParams::default()
        .with_n_threads(std::thread::available_parallelism()?.get() as u32)
        .with_embedding(true)
        .with_seed(1234);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let n_ctx_train = model.n_ctx_train();
    let n_ctx = ctx.n_ctx();

    println!("N_CTX_TRAIN: {n_ctx_train} - N_CTX: {n_ctx}");

    if n_ctx > n_ctx_train as u32 {
        anyhow::bail!("N_CTX is greater than the training context size!");
    }

    unsafe {
        let p = CStr::from_ptr(llama_cpp_sys_2::llama_print_system_info());
        println!("System Info: {:?}", p)
    }

    let mut batch = LlamaBatch::new(512, n_ctx as i32);
    let mut data = Vec::new();
    let normalise = true;

    let mut decode_batch = |sizes: &[usize], batch: &mut LlamaBatch| unsafe {
        ctx.clear_kv_cache();
        let _ = ctx.decode(batch);
        batch.clear();

        for (i, &size) in sizes.iter().enumerate() {
            let embedding = ctx
                .embeddings_ith(i as i32)
                .expect("Embeddings are always enabled");
            // Either normalise immediately, or perform mean-pooling.
            let embedding = if normalise {
                normalize(embedding)
            } else {
                embedding.iter().map(|x| x / size as f32).collect()
            };

            data.push(embedding);
        }
    };

    let mut total_tokens = 0;
    let mut t_batch = 0;
    let mut s_sizes = Vec::new();

    for text in [params.prompt] {
        let mut tokens = model.str_to_token(&text, AddBos::Always)?;

        print_tokens(&model, &tokens);
        // Force the prompt to be at most the size of the context
        tokens.truncate(n_ctx as usize);
        let n_tokens = tokens.len();
        total_tokens += n_tokens;

        // Batch has been filled up
        if t_batch + n_tokens > n_ctx as usize {
            decode_batch(&s_sizes, &mut batch);
            t_batch = 0;
            s_sizes.clear();
        }

        batch.add_sequence(&tokens, s_sizes.len() as i32, false)?;
        t_batch += n_tokens;
        s_sizes.push(n_tokens);
    }

    // Handle last batch
    decode_batch(&s_sizes, &mut batch);
    for (i, embedding) in data.iter().enumerate() {
        println!("Embedding {i}:\n  {embedding:?}\n")
    }

    unsafe { llama_cpp_sys_2::llama_print_timings(ctx.raw_ctx().as_ptr()) }

    Ok(())
}

pub fn normalize(vec: &[f32]) -> Vec<f32> {
    let magnitude = vec
        .iter()
        .fold(0.0, |acc, &val| val.mul_add(val, acc))
        .sqrt();

    if magnitude > f32::EPSILON {
        vec.iter().map(|&val| val / magnitude).collect()
    } else {
        vec.to_vec()
    }
}

fn print_tokens(model: &LlamaModel, tokens: &[LlamaToken]) {
    println!("Number of tokens: {}", tokens.len());
    for token in tokens {
        println!(
            "{:<6} -> {}",
            token.0,
            model.token_to_str(*token).unwrap(),
            // width = 20
        );
    }
}
