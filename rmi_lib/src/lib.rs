mod codegen;
mod models;
mod train;

pub mod optimizer;
pub use models::{RMITrainingData, RMITrainingDataIteratorProvider, ModelInput};
pub use models::KeyType;
pub use optimizer::find_pareto_efficient_configs;
pub use train::{train, train_for_size};
pub use codegen::rmi_size;
pub use codegen::output_rmi;
