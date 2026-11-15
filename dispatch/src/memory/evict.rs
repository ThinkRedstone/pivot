//! Taking ring slots back from operators that keep buffers outside the caches.
//!
//! The caches give slots up through the CLOCK sweep. Operators that hold ring
//! slots of their own, such as a group-by's partition buffers, are outside that
//! sweep, so each dataflow registers the operators that can give slots back in
//! a [`DataflowEvict`] with the worker's memory context. The context turns to
//! those only after the caches have swept their whole region and found every
//! slot pinned.

use crate::graph::Graph;
use crate::memory::WriteBuffer;
use std::convert::Infallible;
use std::ops::ControlFlow;

/// Something that keeps ring slots resident and can give one back on request.
///
/// An operator that holds buffers outside the caches returns one of these from
/// its build (see [`Built::with_evictor`](crate::operations::Built::with_evictor)). The
/// memory context knows nothing about what the buffers contain or how their
/// owner made them safe to release; it only asks for a slot when it has none
/// left to hand out. `evict` runs inside the context's own eviction, so it must
/// only release memory the operator already holds, never allocate or register.
pub trait Evict {
    /// Give up one slot that is no longer needed in memory, or `None` when
    /// every slot this operator keeps is still in use.
    fn evict(&self) -> Option<WriteBuffer>;
}

/// One dataflow's evictors, laid out like its operator graph so that eviction
/// asks operators in a fixed order: the output end of the graph first, then
/// upstream through each operator's publishers.
pub struct DataflowEvict {
    graph: Graph<Option<Box<dyn Evict>>>,
}

impl DataflowEvict {
    /// `evictors[i]` is operator `i`'s evictor, or `None` for an operator that
    /// keeps nothing of its own. `publisher_to_subscriber` are the operator
    /// graph's edges, so the two graphs have the same shape.
    pub fn new(
        evictors: Vec<Option<Box<dyn Evict>>>,
        publisher_to_subscriber: &[(usize, usize)],
    ) -> Self {
        Self {
            graph: Graph::from_edges(evictors, publisher_to_subscriber),
        }
    }

    /// Whether no operator in this dataflow can give slots back, in which case
    /// there is nothing to register.
    pub fn is_empty(&self) -> bool {
        self.graph.nodes().iter().all(Option::is_none)
    }

    /// Forget operator `operator_idx`'s evictor: the operator is gone, and
    /// whatever it kept has been released with it.
    pub fn remove_operator(&mut self, operator_idx: usize) {
        *self.graph.node_mut(operator_idx) = None;
    }

    /// Ask the operators in order for a slot and return the first one offered,
    /// or `None` when no operator has a slot to spare right now.
    pub fn evict(&mut self) -> Option<WriteBuffer> {
        let Ok(outcome) = self.graph.traverse_backwards(|_, evictor| {
            let offered = evictor.as_ref().and_then(|evictor| evictor.evict());
            Ok::<_, Infallible>(match offered {
                Some(buffer) => ControlFlow::Break(buffer),
                None => ControlFlow::Continue(()),
            })
        });
        match outcome {
            ControlFlow::Break(buffer) => Some(buffer),
            ControlFlow::Continue(()) => None,
        }
    }
}

/// An evictor that hands over one buffer the test gave it, once, and notes on
/// `log` that it was asked.
#[cfg(test)]
pub(crate) struct HandOverEvictor {
    pub label: &'static str,
    pub log: std::rc::Rc<std::cell::RefCell<Vec<&'static str>>>,
    pub buffer: std::cell::RefCell<Option<WriteBuffer>>,
}

#[cfg(test)]
impl Evict for HandOverEvictor {
    fn evict(&self) -> Option<WriteBuffer> {
        self.log.borrow_mut().push(self.label);
        self.buffer.borrow_mut().take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{init_test_free_pool, memory_ctx};
    use std::cell::RefCell;
    use std::rc::Rc;

    fn evictor(
        label: &'static str,
        log: &Rc<RefCell<Vec<&'static str>>>,
        buffer: Option<WriteBuffer>,
    ) -> Option<Box<dyn Evict>> {
        Some(Box::new(HandOverEvictor {
            label,
            log: log.clone(),
            buffer: RefCell::new(buffer),
        }))
    }

    #[test]
    fn asks_the_output_end_of_the_graph_first() {
        init_test_free_pool(2);
        let log = Rc::new(RefCell::new(Vec::new()));
        let scan_buffer = memory_ctx().get_write_buffer(false);
        let output_buffer = memory_ctx().get_write_buffer(false);
        let output_slot = output_buffer.slot_idx;
        let mut evictors = DataflowEvict::new(
            vec![
                evictor("scan", &log, Some(scan_buffer)),
                None,
                evictor("output", &log, Some(output_buffer)),
            ],
            &[(0, 1), (1, 2)],
        );

        let evicted = evictors.evict();

        assert_eq!(evicted.map(|buffer| buffer.slot_idx), Some(output_slot));
        assert_eq!(*log.borrow(), ["output"]);
    }

    #[test]
    fn moves_upstream_past_operators_with_nothing_to_give() {
        init_test_free_pool(1);
        let log = Rc::new(RefCell::new(Vec::new()));
        let scan_buffer = memory_ctx().get_write_buffer(false);
        let mut evictors = DataflowEvict::new(
            vec![
                evictor("scan", &log, Some(scan_buffer)),
                evictor("output", &log, None),
            ],
            &[(0, 1)],
        );

        let evicted = evictors.evict();

        assert!(evicted.is_some());
        assert_eq!(*log.borrow(), ["output", "scan"]);
    }

    #[test]
    fn a_dataflow_without_evictors_is_empty() {
        let evictors = DataflowEvict::new(vec![None, None], &[(0, 1)]);

        assert!(evictors.is_empty());
    }
}
