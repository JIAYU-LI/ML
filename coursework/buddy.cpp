/*
 * Buddy Page Allocation Algorithm
 * SKELETON IMPLEMENTATION -- TO BE FILLED IN FOR TASK (2)
 */

/*
 * STUDENT NUMBER: s
 */
#include <infos/mm/page-allocator.h>
#include <infos/mm/mm.h>
#include <infos/kernel/kernel.h>
#include <infos/kernel/log.h>
#include <infos/util/math.h>
#include <infos/util/printf.h>

using namespace infos::kernel;
using namespace infos::mm;
using namespace infos::util;

#define MAX_ORDER	17

/**
 * A buddy page allocation algorithm.
 */
class BuddyPageAllocator : public PageAllocatorAlgorithm
{
private:
	/**
	 * Returns the number of pages that comprise a 'block', in a given order.
	 * @param order The order to base the calculation off of.
	 * @return Returns the number of pages in a block, in the order.
	 */
	static inline constexpr uint64_t pages_per_block(int order)
	{
		/* The number of pages per block in a given order is simply 1, shifted left by the order number.
		 * For example, in order-2, there are (1 << 2) == 4 pages in each block.
		 */
		return (1 << order);
	}

	/**
	 * Returns TRUE if the supplied page descriptor is correctly aligned for the
	 * given order.  Returns FALSE otherwise.
	 * @param pgd The page descriptor to test alignment for.
	 * @param order The order to use for calculations.
	 */
	static inline bool is_correct_alignment_for_order(const PageDescriptor *pgd, int order)
	{
		// Calculate the page-frame-number for the page descriptor, and return TRUE if
		// it divides evenly into the number pages in a block of the given order.
		return (sys.mm().pgalloc().pgd_to_pfn(pgd) % pages_per_block(order)) == 0;
	}

	/** Given a page descriptor, and an order, returns the buddy PGD.  The buddy could either be
	 * to the left or the right of PGD, in the given order.
	 * @param pgd The page descriptor to find the buddy for.
	 * @param order The order in which the page descriptor lives.
	 * @return Returns the buddy of the given page descriptor, in the given order.
	 */
	PageDescriptor *buddy_of(PageDescriptor *pgd, int order)
	{
		// (1) Make sure 'order' is within range
		if (order >= MAX_ORDER) {
			return NULL;
		}

		// (2) Check to make sure that PGD is correctly aligned in the order
		if (!is_correct_alignment_for_order(pgd, order)) {
			return NULL;
		}

		// (3) Calculate the page-frame-number of the buddy of this page.
		// * If the PFN is aligned to the next order, then the buddy is the next block in THIS order.
		// * If it's not aligned, then the buddy must be the previous block in THIS order.
		uint64_t buddy_pfn = is_correct_alignment_for_order(pgd, order + 1) ?
			sys.mm().pgalloc().pgd_to_pfn(pgd) + pages_per_block(order) :
			sys.mm().pgalloc().pgd_to_pfn(pgd) - pages_per_block(order);

		// (4) Return the page descriptor associated with the buddy page-frame-number.
		return sys.mm().pgalloc().pfn_to_pgd(buddy_pfn);
	}

	/**
	 * Inserts a block into the free list of the given order.  The block is inserted in ascending order.
	 * @param pgd The page descriptor of the block to insert.
	 * @param order The order in which to insert the block.
	 * @return Returns the slot (i.e. a pointer to the pointer that points to the block) that the block
	 * was inserted into.
	 */
	PageDescriptor **insert_block(PageDescriptor *pgd, int order)
	{
		// Starting from the _free_area array, find the slot in which the page descriptor
		// should be inserted.
		PageDescriptor **slot = &_free_areas[order];

		// Iterate whilst there is a slot, and whilst the page descriptor pointer is numerically
		// greater than what the slot is pointing to.
		while (*slot && pgd > *slot) {
			slot = &(*slot)->next_free;
		}

		// Insert the page descriptor into the linked list.
		pgd->next_free = *slot;
		*slot = pgd;

		// Return the insert point (i.e. slot)
		return slot;
	}

	/**
	 * Removes a block from the free list of the given order.  The block MUST be present in the free-list, otherwise
	 * the system will panic.
	 * @param pgd The page descriptor of the block to remove.
	 * @param order The order in which to remove the block from.
	 */
	void remove_block(PageDescriptor *pgd, int order)
	{
		// Starting from the _free_area array, iterate until the block has been located in the linked-list.
		PageDescriptor **slot = &_free_areas[order];
		while (*slot && pgd != *slot) {
			slot = &(*slot)->next_free;
		}

		// Make sure the block actually exists.  Panic the system if it does not.
		assert(*slot == pgd);

		// Remove the block from the free list.
		*slot = pgd->next_free;
		pgd->next_free = NULL;
	}

	/**
	 * Given a pointer to a block of free memory in the order "source_order", this function will
	 * split the block in half, and insert it into the order below.
	 * @param block_pointer A pointer to a pointer containing the beginning of a block of free memory.
	 * @param source_order The order in which the block of free memory exists.  Naturally,
	 * the split will insert the two new blocks into the order below.
	 * @return Returns the left-hand-side of the new block.
	 */
	PageDescriptor *split_block(PageDescriptor **block_pointer, int source_order)
	{
		// Make sure there is an incoming pointer.
		assert(*block_pointer);

		// Make sure the block_pointer is correctly aligned.
		assert(is_correct_alignment_for_order(*block_pointer, source_order));

		// Remove the block of free memory in the order "source_order" from the free list of the given order at first.
		remove_block(*block_pointer, source_order);

         	// Then insert the two new blicks into the order below through using insert_block twice.

	 	PageDescriptor **slot1 = insert_block(*block_pointer, source_order-1);

	  	PageDescriptor **slot2 = insert_block(*block_pointer+pages_per_block(source_order-1), source_order-1);

	  	(*slot1)->next_free = *slot2;
		//Returns the left-hand-side of the new block.
	      	return 	*slot1;

	}

	/**
	 * Takes a block in the given source order, and merges it (and it's buddy) into the next order.
	 * This function assumes both the source block and the buddy block are in the free list for the
	 * source order.  If they aren't this function will panic the system.
	 * @param block_pointer A pointer to a pointer containing a block in the pair to merge.
	 * @param source_order The order in which the pair of blocks live.
	 * @return Returns the new slot that points to the merged block.
	 */
	PageDescriptor **merge_block(PageDescriptor **block_pointer, int source_order)
	{
		assert(*block_pointer);

		// Make sure the area_pointer is correctly aligned.
		assert(is_correct_alignment_for_order(*block_pointer, source_order));

		// @param buddy_pointer: A pointer to the buddy block of the pointer block_pointer.
		PageDescriptor *buddy_pointer = buddy_of(*block_pointer, source_order);

   		// Remove the buddy block at first then remove the block in the given source order from relevant free list AND debug them after remove operation which shows that the remove order is important here.
		remove_block(buddy_pointer, source_order);
		//mm_log.messagef(LogLevel::DEBUG, "buddy_pointer %p", sys.mm().pgalloc().pgd_to_pfn(buddy_pointer));
		remove_block(*block_pointer, source_order);
		//mm_log.messagef(LogLevel::DEBUG, "block_pointer %p", sys.mm().pgalloc().pgd_to_pfn(*block_pointer));
		
		// Make sure the merge operation won't be used for the MAX_ORDER.
		if (source_order+1 < MAX_ORDER){
			// Finding the pointer with smaller PFN.
			PageDescriptor *small_pointer = *block_pointer;

			if (*block_pointer > buddy_pointer){
				small_pointer = buddy_pointer;
				}
			// Debug which pointer will be inserted.
			//mm_log.messagef(LogLevel::DEBUG, "small_pointer %p", sys.mm().pgalloc().pgd_to_pfn(small_pointer));

				// Insert the merged block into the order above and return the new slot.
				PageDescriptor **new_slot = insert_block(small_pointer, source_order+1);
				//mm_log.messagef(LogLevel::DEBUG, "insert %p to %d",sys.mm().pgalloc().pgd_to_pfn(*new_slot),(source_order+1));
				return new_slot;

		}
	}

public:
	/**
	 * Constructs a new instance of the Buddy Page Allocator.
	 */
	BuddyPageAllocator() {
		// Iterate over each free area, and clear it.
		for (unsigned int i = 0; i < ARRAY_SIZE(_free_areas); i++) {
			_free_areas[i] = NULL;
		}
	}

	/**
	 * Allocates 2^order number of contiguous pages
	 * @param order The power of two, of the number of contiguous pages to allocate.
	 * @return Returns a pointer to the first page descriptor for the newly allocated page range, or NULL if
	 * allocation failed.
	 */
	PageDescriptor *alloc_pages(int order) override
	{
		// Iterate from the current order to MAX_ORDER in order to find free area that can be allocated.
		for(unsigned int i = order; i < MAX_ORDER; i++){

			// @param order_pointer: A pointer to free area at order i.
	    		PageDescriptor *order_pointer = _free_areas[i];

			// Jugde whether there is free area that can be allocated through checking what the order_pointer points is NULL.
	   		if (order_pointer != NULL){

				// Two cases: free area at the given order cannot be found at the current order so as to interate to the order above or it can be allocated directly at the given order.
	      			if (i != order){

					// After finding the free block, using split function to divide it into required size.
	        			for (int j = i; j > order; j--){
	          				order_pointer = split_block(&order_pointer, j);

	        				}

					// After finding the needed block, remove it from the free list at the given order.
	        			remove_block(order_pointer, order);

	        			return order_pointer;
	    				 }

	    			else {

	     		 		remove_block(order_pointer, order);

	      				return order_pointer;
	    				}
	    			break;
	   			}

	  		}
			// Allocation failed.
	 		return NULL;
	}

	/**
	 * Frees 2^order contiguous pages.
	 * @param pgd A pointer to an array of page descriptors to be freed.
	 * @param order The power of two number of contiguous pages to free.
	 */
	void free_pages(PageDescriptor *pgd, int order) override
	{
		// Make sure that the incoming page descriptor is correctly aligned
		// for the order on which it is being freed, for example, it is
		// illegal to free page 1 in order-1.
		assert(is_correct_alignment_for_order(pgd, order));

		// Insert the page descriptors that need to be freed.
		PageDescriptor *pointer = *insert_block(pgd, order);

		// Iterate from the given order to MAX_ORDER-1 to merge freed pages and their buddy pages.
		for(unsigned int order2 = order; order2 < MAX_ORDER-1; order2++){

			//param pointer2: A pointer to the next block of the page descriptor that need to be freed.
		   	PageDescriptor *pointer2 = pointer->next_free;

			//TWO cases: the buddy is the pointer to next block OR previous block.
		 	if (pointer2 == buddy_of(pointer, order2) or (buddy_of(pointer, order2)-> next_free == pointer)){
				//The pointer after merge function remains the same, for th sake of latter iteration.
		     		pointer = *merge_block(&pointer, order2);

		     		}

		   	else{
		     	
       				// Failing to find its buddy so merge cannot be done, jump out from the order itration.
		    		 break;
		     		}

		  	}
	}

	/**
	 * Reserves a specific page, so that it cannot be allocated.
	 * @param pgd The page descriptor of the page to reserve.
	 * @return Returns TRUE if the reservation was successful, FALSE otherwise.
	 */
	bool reserve_page(PageDescriptor *pgd)
	{
		//mm_log.messagef(LogLevel::DEBUG, "page that need to be reserved %p",sys.mm().pgalloc().pgd_to_pfn(pgd));
		// param order_pointer: A pointer to free area at MAX_ORDER-1 at first.
   		PageDescriptor *order_pointer = _free_areas[MAX_ORDER-1];

		// Iterate from MAX_ORDER-1 to order 0 in order to find the page that need to be reserved.
       		for(unsigned int order = MAX_ORDER-1; order >0; order--){
         	              	while (pgd >= order_pointer and order_pointer != NULL){
					//TWO cases: pgd in the left block or the next block.
                			if (pgd < (order_pointer+pages_per_block(order))){
                  				order_pointer = split_block(&order_pointer, order);
                  				// Jumping out from the while loop so as to split the block that pgs in further to the order below.
                  				break;
                				}
                			else{
                  			//mm_log.messagef(LogLevel::DEBUG, "pgd not in block %p",sys.mm().pgalloc().pgd_to_pfn(order_pointer));
                  				if(order_pointer->next_free != NULL){
                    					order_pointer = order_pointer->next_free;
                  					}
                  				else{
                    					order_pointer = _free_areas[order-1];
							// jumping out from while loop in order to enter the order below.
                    					break;
                  					}

                				}
              				}
              			order_pointer = _free_areas[order-1];

               			}
		// Finding the page that need to be reserved from blocks in the linked list.
       			while (order_pointer != pgd and order_pointer->next_free != NULL){
         			order_pointer=order_pointer->next_free;
       				}
       				if(order_pointer == pgd){
         				remove_block(order_pointer, 0);
               				return true;
      	 				}
       				else {

         				return false;

       					}
	}

	/**
	 * Initialises the allocation algorithm.
	 * @return Returns TRUE if the algorithm was successfully initialised, FALSE otherwise.
	 */
	bool init(PageDescriptor *page_descriptors, uint64_t nr_page_descriptors) override
	{
		mm_log.messagef(LogLevel::DEBUG, "Buddy Allocator Initialising pd=%p, nr=0x%lx", page_descriptors, nr_page_descriptors);

		// Initialise the free area linked list for the maximum order
 
		PageDescriptor **Max_pointer = &_free_areas[MAX_ORDER-1];
		// Obtain the block numbers in the order 16.
    		int blocks = nr_page_descriptors/pages_per_block(MAX_ORDER-1);
		// Iterate blocks in order to insert page descriptors in forming linked list.
    		for (int i = 0; i < blocks; i++){

     			Max_pointer = insert_block(page_descriptors, MAX_ORDER-1);
     			page_descriptors = page_descriptors + pages_per_block(MAX_ORDER-1);


      		}

      		return true;
	}

	/**
	 * Returns the friendly name of the allocation algorithm, for debugging and selection purposes.
	 */
	const char* name() const override { return "buddy"; }

	/**
	 * Dumps out the current state of the buddy system
	 */
	void dump_state() const override
	{
		// Print out a header, so we can find the output in the logs.
		mm_log.messagef(LogLevel::DEBUG, "BUDDY STATE:");

		// Iterate over each free area.
		for (unsigned int i = 0; i < ARRAY_SIZE(_free_areas); i++) {
			char buffer[256];
			snprintf(buffer, sizeof(buffer), "[%d] ", i);

			// Iterate over each block in the free area.
			PageDescriptor *pg = _free_areas[i];
			while (pg) {
				// Append the PFN of the free block to the output buffer.
				snprintf(buffer, sizeof(buffer), "%s%lx ", buffer, sys.mm().pgalloc().pgd_to_pfn(pg));
				pg = pg->next_free;
			}

			mm_log.messagef(LogLevel::DEBUG, "%s", buffer);
		}
	}


private:
	PageDescriptor *_free_areas[MAX_ORDER];
};

/* --- DO NOT CHANGE ANYTHING BELOW THIS LINE --- */

/*
 * Allocation algorithm registration framework
 */
RegisterPageAllocator(BuddyPageAllocator);
