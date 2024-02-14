//! Representation of an initialized llama backend

use crate::LLamaCppError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::SeqCst;
use llama_cpp_sys_2::ggml_log_level;

/// Representation of an initialized llama backend
/// This is required as a parameter for most llama functions as the backend must be initialized
/// before any llama functions are called. This type is proof of initialization.
#[derive(Eq, PartialEq, Debug)]
pub struct LlamaBackend {}

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

impl LlamaBackend {
    /// Mark the llama backend as initialized
    fn mark_init() -> crate::Result<()> {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(false, true, SeqCst, SeqCst) {
            Ok(_) => Ok(()),
            Err(_) => Err(LLamaCppError::BackendAlreadyInitialized),
        }
    }

    /// Initialize the llama backend (without numa).
    ///
    /// # Examples
    ///
    /// ```
    ///# use llama_cpp_2::llama_backend::LlamaBackend;
    ///# use llama_cpp_2::LLamaCppError;
    ///# use std::error::Error;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///
    ///
    /// let backend = LlamaBackend::init()?;
    /// // the llama backend can only be initialized once
    /// assert_eq!(Err(LLamaCppError::BackendAlreadyInitialized), LlamaBackend::init());
    ///
    ///# Ok(())
    ///# }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn init() -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe { llama_cpp_sys_2::llama_backend_init(false) }
        Ok(LlamaBackend {})
    }

    /// Initialize the llama backend (with numa).
    /// ```
    ///# use llama_cpp_2::llama_backend::LlamaBackend;
    ///# use std::error::Error;
    ///
    ///# fn main() -> Result<(), Box<dyn Error>> {
    ///  let llama_backend = LlamaBackend::init_numa()?;
    ///
    ///# Ok(())
    ///# }
    /// ```
    #[tracing::instrument(skip_all)]
    pub fn init_numa() -> crate::Result<LlamaBackend> {
        Self::mark_init()?;
        unsafe { llama_cpp_sys_2::llama_backend_init(true) }
        Ok(LlamaBackend {})
    }

    /// Change the output of llama.cpp's logging to be voided instead of pushed to `stderr`.
    pub fn void_logs(&mut self) {
        unsafe extern "C" fn void_log(
            _level: ggml_log_level,
            _text: *const ::std::os::raw::c_char,
            _user_data: *mut ::std::os::raw::c_void,
        ) {}

        unsafe {
            unsafe { llama_cpp_sys_2::llama_log_set(Some(void_log), std::ptr::null_mut()) }
        }
    }
}

/// Drops the llama backend.
/// ```
///
///# use llama_cpp_2::llama_backend::LlamaBackend;
///# use std::error::Error;
///
///# fn main() -> Result<(), Box<dyn Error>> {
/// let backend = LlamaBackend::init()?;
/// drop(backend);
/// // can be initialized again after being dropped
/// let backend = LlamaBackend::init()?;
///# Ok(())
///# }
///
/// ```
impl Drop for LlamaBackend {
    fn drop(&mut self) {
        match LLAMA_BACKEND_INITIALIZED.compare_exchange(true, false, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(_) => {
                unreachable!("This should not be reachable as the only ways to obtain a llama backend involve marking the backend as initialized.")
            }
        }
        unsafe { llama_cpp_sys_2::llama_backend_free() }
    }
}
