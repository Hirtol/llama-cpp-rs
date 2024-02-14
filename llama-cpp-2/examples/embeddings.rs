//! This is an translation of simple.cpp in llama.cpp using llama-cpp-2.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use anyhow::{bail, Context, Result};
use clap::Parser;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::ggml_time_us;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::AddBos;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::ffi::CStr;
use std::io::Write;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::slice;
use std::time::Duration;

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
        // .with_n_ctx(NonZeroU32::new(2048))
        .with_n_threads(std::thread::available_parallelism()?.get() as u32)
        .with_embedding(true)
        .with_seed(1234);

    let mut ctx = model
        .new_context(&backend, ctx_params)
        .with_context(|| "unable to create the llama_context")?;

    let n_ctx_train = unsafe { llama_cpp_sys_2::llama_n_ctx_train(model.model.as_ptr()) };
    let n_ctx = ctx.n_ctx();

    println!("N_CTX_TRAIN: {n_ctx_train} - n_ctx: {n_ctx}");

    if n_ctx > n_ctx_train as u32 {
        anyhow::bail!("N_CTX is greater than the training context size!");
    }

    unsafe {
        let p = CStr::from_ptr(llama_cpp_sys_2::llama_print_system_info());
        println!("System Info: {:?}", p)
    }

    let mut tokens = model.str_to_token(&params.prompt, AddBos::Always)?;
    println!("Number of tokens: {}", tokens.len());
    for token in &tokens {
        println!("{}", model.token_to_str(*token)?);
    }

    if tokens.len() > n_ctx as usize {
        anyhow::bail!(
            "Prompt (`{}`) is longer than context window (`{n_ctx}`)",
            tokens.len()
        )
    }

    let mut batch = LlamaBatch::new(512, 1);

    let mut total_tokens = 0;
    let mut t_batch = 0;
    let mut s_sizes = Vec::new();

    for text in [params.prompt] {
        let mut tokens = model.str_to_token(&text, AddBos::Always)?;

        tokens.truncate(n_ctx as usize);
        let n_tokens = tokens.len();
        total_tokens += n_tokens;

        // Batch has been filled up
        if t_batch + n_tokens > n_ctx as usize {
            t_batch = 0;
        }
    }

    let last_index: i32 = (tokens.len() - 1) as i32;
    for (i, token) in (0_i32..).zip(tokens.into_iter()) {
        // llama_decode will output logits only for the last token of the prompt
        let is_last = i == last_index;
        batch.add(token, i, &[0], is_last)?;
    }
    ctx.decode(&mut batch)
        .with_context(|| "llama_decode() failed")?;

    let n_embd = unsafe { llama_cpp_sys_2::llama_n_embd(model.model.as_ptr()) };
    dbg!(n_embd);

    let embeddings = unsafe { llama_cpp_sys_2::llama_get_embeddings(ctx.context.as_ptr()) };

    let embeddings = unsafe { slice::from_raw_parts(embeddings, n_embd as usize) };

    pub fn normalize(vec: &[f32]) -> Vec<f32> {
        let magnitude = (vec.iter().fold(0.0, |acc, &val| val.mul_add(val, acc))).sqrt();

        if magnitude > f32::EPSILON {
            vec.iter().map(|&val| val / magnitude).collect()
        } else {
            vec.to_vec()
        }
    }
    let norm_embeddings = normalize(embeddings);

    println!("Embeddings: {norm_embeddings:?}");

    // OTHER
    // let mut tokens = model.str_to_token("world", AddBos::Always)?;
    // //unsafe { llama_cpp_sys_2::llama_kv_cache_clear(ctx.context.as_ptr()) };
    // let mut batch = LlamaBatch::new(512, 1);
    //
    // let last_index: i32 = (tokens.len() - 1) as i32;
    // for (i, token) in (0_i32..).zip(tokens.into_iter()) {
    //     // llama_decode will output logits only for the last token of the prompt
    //     let is_last = i == last_index;
    //     batch.add(token, i, &[0], is_last)?;
    // }
    // ctx.decode(&mut batch)
    //     .with_context(|| "llama_decode() failed")?;
    //
    // let n_embd = unsafe { llama_cpp_sys_2::llama_n_embd(model.model.as_ptr()) };
    // dbg!(n_embd);
    //
    // let embeddings = unsafe { llama_cpp_sys_2::llama_get_embeddings(ctx.context.as_ptr()) };
    //
    // let embeddings = unsafe { slice::from_raw_parts(embeddings, n_embd as usize) };
    //
    // let norm_embeddings = normalize(embeddings);
    //
    // println!("Embeddings: {norm_embeddings:?}");

    Ok(())
}
