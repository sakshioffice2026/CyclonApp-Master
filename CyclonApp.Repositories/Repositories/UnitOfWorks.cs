using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;
using CyclonApp.Repositories.Contracts;
using Microsoft.AspNetCore.Http;

namespace CyclonApp.Repositories.Repositories
{
    public class UnitOfWorks : IUnitOfWork
    {
        private readonly ApplicationDbContext _Dbcontext;
        private readonly IHttpContextAccessor _httpContextAccessor;

        public UnitOfWorks(
            ApplicationDbContext dbContext,
            IHttpContextAccessor httpContextAccessor)
        {
            _Dbcontext = dbContext;
            _httpContextAccessor = httpContextAccessor;

            exceptionHandlerRepository =
                new ExceptionHandlerRepository(_Dbcontext);
        }

        public ExceptionHandlerRepository exceptionHandlerRepository { get; private set; }

        public async Task Commit()
        {
            await _Dbcontext.SaveChangesAsync();
        }

        public void Dispose()
        {
            _Dbcontext.Dispose();
        }
    }
}

